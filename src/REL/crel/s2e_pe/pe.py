# PEMD
from tokenizers.pre_tokenizers import Whitespace

pre_tokenizer = Whitespace()
try:
    import spacy

    nlp = spacy.load("en_core_web_md")
except:  # From Google Colab (see https://stackoverflow.com/a/59197634)
    import spacy.cli

    spacy.cli.download("en_core_web_md")
    import en_core_web_md

    nlp = en_core_web_md.load()
import torch
from transformers import AutoConfig, AutoTokenizer, LongformerConfig

# EEMD
from . import data
from .coref_bucket_batch_sampler import BucketBatchSampler
from .modeling import S2E
from .pe_data import PreProcess  # to use get_span()


class PEMD:
    """Responsible for personal entity mention detection"""

    def __init__(self):
        self.pronouns = ["my", "our"]  # These should be lowercase
        self.preprocess = PreProcess()  # to use get_span()

    def _extract_text_with_pronoun(self, utt: str, max_candidate_num=10):
        """

        Args:
            max_candidate_num (int): Max following words num (which equals to candidate num). Does not contain "my/our" in this count.

        Example:
            Input: 'Our town is big into high school football - our quarterback just left to go play for Clemson. Oh, that is my town.'
            Output:
                [{'extracted_text': 'Our town is big into high school football - our quarterback',
                'pronoun': ('Our', (0, 3))}, ...]
        """
        if any(
            [True for p in self.pronouns if p in utt.lower()]
        ):  # If at least one pronoun is in utt.lower()
            ret = []
        else:  # If no pronouns are in utt.lower()
            return []

        try:  # if tokenizer version is 0.10.3 etc where pre_tokenize_str is available
            ws = pre_tokenizer.pre_tokenize_str(
                utt
            )  # E.g., [('Our', (0, 3)), ('town', (4, 8)), ...]
        except:  # if 0.8.1.rc2 etc where pre_tokenizer_str is NOT available
            ws = pre_tokenizer.pre_tokenize(utt)
        for i, (w, _) in enumerate(ws):
            if w.lower() in self.pronouns:
                n_options = min(
                    max_candidate_num, len(ws[i:]) - 1
                )  # `len(ws[i:])` contains "my/our" so have to operate -1
                text_w_pronoun = utt[
                    ws[i][1][0] : ws[i + n_options][1][1]
                ]  # E.g., 'our quarterback just ...' # `ws[i][1][0]`: start position. `ws[i+n+2][1][1]`: end position.
                ret.append({"pronoun": ws[i], "extracted_text": text_w_pronoun})
        return ret

    def pem_detector(self, utt):
        """Mention detection for personal entities

        Args:
            utt (str): Input utterance.
                E.g., 'I agree. One of my favorite forms of science fiction is anything related to time travel! I find it fascinating.'

        Returns:
            list of detected personal entity mentions.
                E.g., ['my favorite forms of science fiction']

        """
        _dc_list = self._extract_text_with_pronoun(
            utt
        )  # E.g., [{'extracted_text': 'Our town is big into high ...', 'pronoun': ('Our', (0, 3))}, ...]
        if len(_dc_list) == 0:
            return []
        else:
            texts_w_pronoun = [
                _dc["extracted_text"] for _dc in _dc_list
            ]  # E.g., ['Our town is big into ...', 'My dog loves human food!']

        ret = []
        for text in texts_w_pronoun:  # For each extracted text
            doc = nlp(text)
            ment = ""
            end_pos = 0  # start_pos is always 0
            for i, token in enumerate(doc):
                # print(token.pos_, token.text)
                if i == 0:
                    assert (
                        token.text.lower() in self.pronouns
                    ), f"{token.text} does not start with 'my' or 'our'"
                    end_pos = token.idx + len(token.text)  # update end_pos
                else:  # i > 0
                    if token.pos_ in [
                        "ADJ",
                        "NOUN",
                        "PROPN",
                        "NUM",
                        "PART",
                    ] or token.text.lower() in [
                        "of",
                        "in",
                        "the",
                        "a",
                        "an",
                    ]:
                        end_pos = token.idx + len(token.text)  # update end_pos
                    else:
                        break
            ment = text[:end_pos]

            ###### Post process #######
            # if end with " of " then remove it
            for drop_term in [" of", " in", " to"]:
                ment = (
                    ment[: -(len(drop_term) - 1)] if ment.endswith(drop_term) else ment
                )

            if (
                len(ment) > min([len(pron) for pron in self.pronouns]) + 1
            ):  # Want to ignore the case: "My "
                ret.append(ment.strip())

            # 220406: TMP error check
            # TODO: Check this part whether it is needed or not
            assert len(ment) != "our ", f'Should fix "if len(ment)>len(CLUE)+1" part.'

            # Sanity check
            for ment in ret:
                assert ment in utt, f"{ment} is not in {utt}"

        # Change to REL format [start_pos, length, mention]
        ret = [
            [start_pos, len(m), m]
            for m in ret
            for start_pos, _ in self.preprocess.get_span(
                m, utt, flag_start_end_span_representation=False
            )
        ]

        return ret


class EEMD:
    """Find corresponding explicit entity mention using s2e-coref-based method"""

    def __init__(self, s2e_pe_model):
        self.conf = self.Config(s2e_pe_model=s2e_pe_model)
        self.model = self._read_model()

    class Config:
        """Inner class for config"""

        def __init__(self, s2e_pe_model):
            self.max_seq_length = 4096
            self.model_name_or_path = s2e_pe_model
            self.max_total_seq_len = 4096
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.device = torch.device("cpu") # TMP

            # Config for NN model
            # Params are from: https://github.com/yuvalkirstain/s2e-coref/blob/main/cli.py
            self.max_span_length = 30
            self.top_lambda = 0.4
            self.ffnn_size = 3072
            self.normalise_loss = False
            self.dropout_prob = 0.3

    def _read_model(self):
        config_class = LongformerConfig
        base_model_prefix = "longformer"

        transformer_config = AutoConfig.from_pretrained(
            self.conf.model_name_or_path
        )  # , cache_dir=args.cache_dir)

        S2E.config_class = config_class
        S2E.base_model_prefix = base_model_prefix

        model = S2E.from_pretrained(
            self.conf.model_name_or_path, config=transformer_config, args=self.conf
        )

        model.to(self.conf.device)

        return model

    def get_scores(self, input_data):
        """Calculate the score of each mention pair
        Args:
            input_data (dict): Input data.
                E.g., {'clusters': [], # Not used for inference
                       'doc_key': 'tmp', # Not used for inference
                       'mentions': [[2, 3], [77, 78]], # Detected concept and NE mentions
                       'pems': [[67, 72]], # Detected personal entity mention. Only one PEM is allowed now.
                       'sentences': [['I', 'think', 'science', 'fiction', 'is', ...]], # tokenized sentences using tokenizers.pre_tokenizers
                       'speakers': [['USER', 'USER', 'USER', 'USER', 'USER', ...]], # speaker information
                       'text': None
                       }

        Returns:
            The scores for each mention pair. The pairs which does not have any PEM are removed in the later post-processing.
                E.g.,
                    [{'doc_key': 'tmp',
                    'span_token_anaphora': (67, 72),
                    'span_token_antecedent': (2, 3), ...]
        """
        assert (
            type(input_data) == dict
        ), f"input_data should be a dict, but got {type(input_data)}"
        tokenizer = AutoTokenizer.from_pretrained(
            self.conf.model_name_or_path, use_fast=False
        )
        # `use_fast=False` should be supecified for v4.18.0 (do not need to do this for v3.3.1)
        # See: https://github.com/huggingface/transformers/issues/10297#issuecomment-812548803

        eval_dataset = data.get_dataset(tokenizer, input_data, self.conf)

        eval_dataloader = BucketBatchSampler(
            eval_dataset,
            max_total_seq_len=self.conf.max_total_seq_len,
            batch_size_1=True,
        )

        assert len(eval_dataloader) == 1, f"Currently, only 1 batch is supported"
        for i, ((doc_key, subtoken_maps), batch) in enumerate(eval_dataloader):
            # NOTE: subtoken_maps should NOT be used to map word -> subtoken!!!
            #       The original name of subtoken_maps is `end_token_idx_to_word_idx`, meaning this is intended to map subtoken (end token) -> word.
            # NOTE: Currently, this code only supports only one example at a time, however, for the futurework, we keep this for loop here.
            batch = tuple(tensor.to(self.conf.device) for tensor in batch)
            input_ids, attention_mask, gold_clusters = batch

            with torch.no_grad():
                _, scores = self.model(
                    input_ids=input_ids,  # This calls __call__ in module.py in PyTorch, and it calls S2E.forward().
                    attention_mask=attention_mask,
                    gold_clusters=gold_clusters,
                    return_all_outputs=True,
                    subtoken_map=subtoken_maps,  # HJ
                    pem_eem_subtokenspan=(
                        sorted(
                            eval_dataset.dockey2eems_subtokenspan[doc_key]
                            + eval_dataset.dockey2pems_subtokenspan[doc_key]
                        )
                    ),  # HJ
                    doc_key=doc_key,
                )  # HJ: 220221

        return scores
