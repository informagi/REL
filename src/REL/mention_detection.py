import re
import sys
from termcolor import colored
from flair.data import Sentence
from flair.models import SequenceTagger
from syntok import segmenter

from REL.mention_detection_base import MentionDetectionBase



class MentionDetection(MentionDetectionBase):
    """
    Class responsible for mention detection.
    """
    def __init__(self, base_url, wiki_version):
        self.cnt_exact = 0
        self.cnt_partial = 0
        self.cnt_total = 0

        super().__init__(base_url, wiki_version)

    def format_spans(self, dataset):
        """
        Responsible for formatting given spans into dataset for the ED step. More specifically,
        it returns the mention, its left/right context and a set of candidates.

        :return: Dictionary with mentions per document.
        """

        dataset, _, _ = self.split_text(dataset)
        results = {}
        total_ment = 0

        for doc in dataset:
            contents = dataset[doc]
            sentences_doc = [v[0] for v in contents.values()]

            results_doc = []
            for idx_sent, (sentence, spans) in contents.items():
                for ngram, start_pos, end_pos in spans:
                    total_ment += 1

                    # end_pos = start_pos + length
                    # ngram = text[start_pos:end_pos]
                    mention = self.preprocess_mention(ngram)
                    left_ctxt, right_ctxt = self.get_ctxt(
                        start_pos, end_pos, idx_sent, sentence, sentences_doc
                    )

                    chosen_cands = self.get_candidates(mention)
                    res = {
                        "mention": mention,
                        "context": (left_ctxt, right_ctxt),
                        "candidates": chosen_cands,
                        "gold": ["NONE"],
                        "pos": start_pos,
                        "sent_idx": idx_sent,
                        "ngram": ngram,
                        "end_pos": end_pos,
                        "sentence": sentence,
                    }

                    results_doc.append(res)
            results[doc] = results_doc
        return results, total_ment


    def split_single(self, text):
        sentences_as_token_lists = segmenter.analyze(text)
        sentences = []
        for paragraph in sentences_as_token_lists:
            for sentence in paragraph:
                tokens = [ str(token) for token in sentence ]
                sentences.append("".join(tokens))
        return sentences


    def split_text(self, dataset, process_sentences, split_docs_value=0, tagger=None, is_flair=False):
        """
        Splits text into sentences with optional spans (format is a requirement for GERBIL usage).
        This behavior is required for the default NER-tagger, which during experiments was experienced
        to achieve higher performance.

        :return: dictionary with sentences and optional given spans per sentence.
        """

        res = {}
        splits = [0]
        processed_sentences = []
        for doc in dataset:
            text, spans = dataset[doc]
            if process_sentences:
                sentences = self.split_single(text)
            elif split_docs_value > 0:
                sentences = self.split_text_in_parts(text, split_docs_value, tagger)
            else:
                sentences = [ text ]
            res[doc] = {}

            i = 0
            pos_end = 0  # Added  (issue #49)
            for sent in sentences:
                if len(sent.strip()) == 0:
                    continue
                # Match gt to sentence.
                # pos_start = text.find(sent) # Commented out (issue #49)
                pos_start = text.find(sent, pos_end)  # Added  (issue #49)
                pos_end = pos_start + len(sent)

                # ngram, start_pos, end_pos
                spans_sent = [
                    [text[x[0] : x[0] + x[1]], x[0], x[0] + x[1]]
                    for x in spans
                    if pos_start <= x[0] < pos_end
                ]
                res[doc][i] = [sent, spans_sent]
                if len(spans) == 0:
                    processed_sentences.append(
                        Sentence(sent, use_tokenizer=True) if is_flair else sent
                    )
                i += 1
            splits.append(splits[-1] + i)
        return res, processed_sentences, splits


    def combine_entities(self, ner_results):
        ner_results_out = []
        i = 0
        while i < len(ner_results)-1:
            last_end = ner_results[i]["end"]
            ner_results_out.append(dict(ner_results[i]))
            j = 1
            while i + j < len(ner_results) and (ner_results[i+j]["start"] == last_end or
                                                (ner_results[i+j]["start"] == last_end + 1 and 
                                                 re.search("^I", ner_results[i+j]["entity"]) and
                                                 re.sub("^..", "", ner_results[i+j]["entity"]) == re.sub("^..", "", ner_results[i]["entity"]))):
                if ner_results[i+j]["start"] == last_end:
                    ner_results_out[-1]["word"] += re.sub("^##", "", ner_results[i+j]["word"])
                else:
                    ner_results_out[-1]["word"] += " " + ner_results[i+j]["word"]
                ner_results_out[-1]["end"] = ner_results[i+j]["end"]
                last_end = ner_results[i+j]["end"]
                j += 1
            i += j
        return ner_results_out


    def split_sentence_in_bert_tokens(self, sentence, tagger):
        tokenizer_results = tagger.tokenizer([sentence], return_offsets_mapping=True) # warns if sentence is too long (>512)
        input_ids = tokenizer_results["input_ids"][0]
        token_spans = tokenizer_results["offset_mapping"][0]
        tokens = [ tagger.tokenizer.decode(token_id) for token_id in input_ids ]
        return tokens, token_spans



    def combine_tokens_to_text(self, token_list):
        text = ""
        for token in token_list:
            if re.search("^##", token):
                text += re.sub("^##", "", token)
            elif text == "":
                text = token
            else:
                text += " " + token
        return text


    def split_text_in_parts(self, text, split_docs_value, tagger):
        """
        Splits text in parts of as most split_docs_value tokens. Texts are split at sentence 
        boundaries. If a sentence is longer than the limit it will be split in parts of
        maximally split_docs_value tokens.
        """
        sentences = self.split_single(text)
        token_lists = []
        texts = []
        for sentence in sentences:
            sentence_tokens, token_spans = self.split_sentence_in_bert_tokens(sentence, tagger)
            if len(token_lists) == 0 or (len(token_lists[-1]) + len(sentence_tokens)) > split_docs_value:
                token_lists.append([])
                texts.append("")
            token_lists[-1].extend(sentence_tokens)
            if texts[-1] == "":
                texts[-1] = sentence
            else:
                texts[-1] += sentence
            first_split_point = 0
            while len(token_lists[-1]) > split_docs_value:
                token_lists.append(list(token_lists[-1]))
                token_lists[-2] = token_lists[-2][:split_docs_value]
                token_lists[-1] = token_lists[-1][split_docs_value:]
                second_split_point = token_spans[-len(token_lists[-1])][0]
                texts[-1] = sentence[first_split_point:second_split_point]
                texts.append(sentence[second_split_point:])
                first_split_point = second_split_point
        return texts


    def find_mentions(self, dataset, use_bert, process_sentences, split_docs_value, tagger=None):
        """
        Responsible for finding mentions given a set of documents in a batch-wise manner. More specifically,
        it returns the mention, its left/right context and a set of candidates.
        :return: Dictionary with mentions per document.
        """
        if tagger is None:
            raise Exception(
                "No NER tagger is set, but you are attempting to perform Mention Detection.."
            )
        # Verify if Flair, else ngram or custom.
        is_flair = isinstance(tagger, SequenceTagger)
        dataset_sentences_raw, processed_sentences, splits = self.split_text(
            dataset, process_sentences, split_docs_value, tagger, is_flair
        )
        results = {}
        total_ment = 0
        if is_flair:
            tagger.predict(processed_sentences)
        for i, doc in enumerate(dataset_sentences_raw):
            raw_text = dataset[doc][0]
            contents = dataset_sentences_raw[doc]
            sentences_doc = [v[0] for v in contents.values()]
            sentences = processed_sentences[splits[i] : splits[i + 1]]
            result_doc = []
            cum_sent_length = 0
            offset = 0
            for (idx_sent, (sentence, ground_truth_sentence)), snt in zip(
                contents.items(), sentences
            ):

                # Only include offset if using Flair.
                # if is_flair:
                # 20220607: no always include
                offset = raw_text.find(sentence, cum_sent_length)
                if offset < 0:
                    print(colored(f"sentence not found in text: cannot happen: {sentence}", "red"), file=sys.stderr)
                    offset = 0
                entity_counter = 0
                for entity in (
                    snt.get_spans("ner")
                    if is_flair
                    else self.combine_entities(tagger(snt))
                ):
                    if use_bert:
                        text, start_pos, end_pos, conf, tag = (
                            sentence[entity["start"]:entity["end"]], # for BERT
                            entity["start"],
                            entity["end"],
                            entity["score"],
                            entity["entity"],
                        )
                    else:
                        text, start_pos, end_pos, conf, tag = (
                            entity.text, # for Flair
                            entity.start_position,
                            entity.end_position,
                            entity.score,
                            entity.tag,
                        )
                    total_ment += 1
                    m = self.preprocess_mention(text)
                    cands = self.get_candidates(m)
                    if len(cands) == 0:
                        continue
                    # Re-create ngram as 'text' is at times changed by Flair (e.g. double spaces are removed).
                    ngram = sentence[start_pos:end_pos]
                    left_ctxt, right_ctxt = self.get_ctxt(
                        start_pos, end_pos, idx_sent, sentence, sentences_doc
                    )
                    res = {
                        "mention": m,
                        "context": (left_ctxt, right_ctxt),
                        "candidates": cands,
                        "gold": ["NONE"],
                        "pos": start_pos + offset,
                        "sent_idx": idx_sent,
                        "ngram": ngram,
                        "end_pos": end_pos + offset,
                        "sentence": sentence,
                        "conf_md": conf,
                        "tag": tag,
                    }
                    result_doc.append(res)
                cum_sent_length += len(sentence) + (offset - cum_sent_length)
            results[doc] = result_doc
        return results, total_ment
