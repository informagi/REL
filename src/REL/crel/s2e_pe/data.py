import json
import logging
import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset

from .consts import NULL_ID_FOR_COREF, SPEAKER_END, SPEAKER_START
from .utils import flatten_list_of_lists

CorefExample = namedtuple("CorefExample", ["token_ids", "clusters"])

logger = logging.getLogger(__name__)


class CorefDataset(Dataset):
    def __init__(self, input_data, tokenizer, model_name_or_path, max_seq_length=-1):
        self.tokenizer = tokenizer
        (
            examples,
            self.max_mention_num,
            self.max_cluster_size,
            self.max_num_clusters,
            dockey2eems_tokenspan,
            dockey2pems_tokenspan,
        ) = self._parse_jsonlines(input_data)
        self.max_seq_length = max_seq_length
        (
            self.examples,
            self.lengths,
            self.num_examples_filtered,
            self.dockey2eems_subtokenspan,
            self.dockey2pems_subtokenspan,
        ) = self._tokenize(
            examples, dockey2eems_tokenspan, dockey2pems_tokenspan, model_name_or_path
        )
        logger.info(
            f"Finished preprocessing Coref dataset. {len(self.examples)} examples were extracted, {self.num_examples_filtered} were filtered due to sequence length."
        )

    def _parse_jsonlines(self, d):
        examples = []
        max_mention_num = -1
        max_cluster_size = -1
        max_num_clusters = -1
        dockey2pems_tokenspan = {}
        dockey2eems_tokenspan = {}
        doc_key = d["doc_key"]
        assert (
            type(d["sentences"][0]) == list
        ), "'sentences' should be 2d list, not just a 1d list of the tokens."
        input_words = flatten_list_of_lists(d["sentences"])
        clusters = d["clusters"]
        max_mention_num = max(max_mention_num, len(flatten_list_of_lists(clusters)))
        max_cluster_size = max(
            max_cluster_size,
            max(len(cluster) for cluster in clusters) if clusters else 0,
        )
        max_num_clusters = max(max_num_clusters, len(clusters) if clusters else 0)
        speakers = flatten_list_of_lists(d["speakers"])
        examples.append((doc_key, input_words, clusters, speakers))
        dockey2eems_tokenspan[doc_key] = d["mentions"]
        dockey2pems_tokenspan[doc_key] = d["pems"]
        return (
            examples,
            max_mention_num,
            max_cluster_size,
            max_num_clusters,
            dockey2eems_tokenspan,
            dockey2pems_tokenspan,
        )

    def _tokenize(
        self, examples, dockey2eems_tokenspan, dockey2pems_tokenspan, model_name_or_path
    ):
        coref_examples = []
        lengths = []
        num_examples_filtered = 0
        dockey2eems_subtokenspan = {}
        dockey2pems_subtokenspan = {}
        for doc_key, words, clusters, speakers in examples:
            word_idx_to_start_token_idx = dict()
            word_idx_to_end_token_idx = dict()
            end_token_idx_to_word_idx = [0]  # for <s>

            token_ids = []
            last_speaker = None
            for idx, (word, speaker) in enumerate(zip(words, speakers)):
                if last_speaker != speaker:
                    speaker_prefix = (
                        [SPEAKER_START]
                        + self.tokenizer.encode(" " + speaker, add_special_tokens=False)
                        + [SPEAKER_END]
                    )
                    last_speaker = speaker
                else:
                    speaker_prefix = []
                for _ in range(len(speaker_prefix)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(speaker_prefix)
                word_idx_to_start_token_idx[idx] = len(token_ids) + 1  # +1 for <s>
                tokenized = self.tokenizer.encode(" " + word, add_special_tokens=False)
                for _ in range(len(tokenized)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(tokenized)
                word_idx_to_end_token_idx[idx] = len(
                    token_ids
                )  # old_seq_len + 1 (for <s>) + len(tokenized_word) - 1 (we start counting from zero) = len(token_ids)

            if 0 < self.max_seq_length < len(token_ids):
                num_examples_filtered += 1
                continue

            new_clusters = [
                [
                    (word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end])
                    for start, end in cluster
                ]
                for cluster in clusters
            ]
            lengths.append(len(token_ids))

            coref_examples.append(
                (
                    (doc_key, end_token_idx_to_word_idx),
                    CorefExample(token_ids=token_ids, clusters=new_clusters),
                )
            )

            dockey2eems_subtokenspan[doc_key] = [
                (word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end])
                for start, end in dockey2eems_tokenspan[doc_key]
            ]
            dockey2pems_subtokenspan[doc_key] = [
                (word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end])
                for start, end in dockey2pems_tokenspan[doc_key]
            ]

        return (
            coref_examples,
            lengths,
            num_examples_filtered,
            dockey2eems_subtokenspan,
            dockey2pems_subtokenspan,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def pad_clusters_inside(self, clusters):
        return [
            cluster
            + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)]
            * (self.max_cluster_size - len(cluster))
            for cluster in clusters
        ]

    def pad_clusters_outside(self, clusters):
        return clusters + [[]] * (self.max_num_clusters - len(clusters))

    def pad_clusters(self, clusters):
        clusters = self.pad_clusters_outside(clusters)
        clusters = self.pad_clusters_inside(clusters)
        return clusters

    def _pe_create_tensored_batch(self, padded_batch, len_example):
        """Create tensored_batch avoiding errors
        Original code was:
        `tensored_batch = tuple(torch.stack([example[i].squeeze() for example in padded_batch], dim=0) for i in range(len(example)))`
        However, this does not handle the single cluster case (E.g., "clusters": [[[135, 136], [273, 273]]] in the train.english.jsonlines)

        The error caused by the above is like (220322):
            gold_clusters = [tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters.tolist()]
            TypeError: argument of type 'int' is not iterable

        - Updates:
          - 220228: Created
          - 220322: Write the error details
        """
        assert len_example == 3
        tensored_batch = tuple()
        for i in range(len_example):
            to_stack = []
            for example in padded_batch:
                assert (
                    len(example) == 3
                ), f"example contains three components: input_ids, attention_mask, and clusters. Current len(examples): {len(example)}"
                if i < 2:  # input_ids and attention_mask
                    to_stack.append(example[i].squeeze())
                elif i == 2:  # clusters
                    to_stack.append(
                        example[i]
                    )  # squeeze() cause the error to single-cluster case
            # add to_stack to tensored_batch (tuple)
            tensored_batch += (torch.stack(to_stack, dim=0),)
        return tensored_batch

    def pad_batch(self, batch, max_length):
        max_length += 2  # we have additional two special tokens <s>, </s>
        padded_batch = []
        for example in batch:
            # encoded_dict = self.tokenizer.encode_plus(example[0], # This does not work transformers v4.18.0 (works with v3.3.1)
            # See: https://github.com/huggingface/transformers/issues/10297
            encoded_dict = self.tokenizer.encode_plus(
                example[0],
                use_fast=False,
                add_special_tokens=True,
                pad_to_max_length=True,
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="pt",
            )
            clusters = self.pad_clusters(example.clusters)
            example = (encoded_dict["input_ids"], encoded_dict["attention_mask"]) + (
                torch.tensor(clusters),
            )
            padded_batch.append(example)
        # tensored_batch = tuple(torch.stack([example[i].squeeze() for example in padded_batch], dim=0) for i in range(len(example)))
        tensored_batch = self._pe_create_tensored_batch(
            padded_batch, len(example)
        )  # HJ: 220228
        return tensored_batch


def get_dataset(tokenizer, input_data, conf):
    """Read input data

    Args:
        - tokenizer
        - input_data (dict): Input dict containing the following keys:
            dict_keys(['clusters', 'doc_key', 'mentions', 'pems', 'sentences', 'speakers'])
            E.g.,
                test_jsonl = {
                    "clusters": [[[78, 83], [88, 89]]], # This can be blank when you want to perform prediction.
                    "doc_key": "dialind:0_turn:3_pem:my-favorite-forms-of-science-fiction", # doc_key should be unique, no restrictions on the format
                    "mentions": [[35, 35], [37, 38], [74, 74], [85, 85], [88, 89]], # mentions and spans should be token-level spans (i.e., different from REL). See original document of s2e-coref.
                    "pems": [[78, 83]],
                    "sentences": [["I", "think", "science", "fiction", "is", ...], ...],
                    "speakers": [["SYSTEM", "SYSTEM", "SYSTEM", ..., "USER", "USER", "USER", ...], ...], }


    Returns:
        - dataset (CorefDataset):

    Notes:
        - Currently, parallel processing is not supported, i.e., you cannot input more than or equal to two sentences or PEMs at the same time.
    """

    coref_dataset = CorefDataset(
        input_data,
        tokenizer,
        max_seq_length=conf.max_seq_length,
        model_name_or_path=conf.model_name_or_path,
    )
    return coref_dataset
