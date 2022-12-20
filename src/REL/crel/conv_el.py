import importlib
import sys
from pathlib import Path

from .bert_md import BERT_MD
from .s2e_pe import pe_data
from .s2e_pe.pe import EEMD, PEMD
from REL.response_model import ResponseModel


class ConvEL:
    def __init__(
        self, base_url=".", wiki_version="wiki_2019", ed_model=None, user_config=None, threshold=0
    ):
        self.threshold = threshold

        self.wiki_version = wiki_version
        self.base_url = base_url
        self.file_pretrained = str(Path(base_url) / "bert_conv-td")

        self.bert_md = BERT_MD(self.file_pretrained)

        if not ed_model:
            ed_model = self._default_ed_model()

        self.response_model = ResponseModel(self.base_url, self.wiki_version, model=ed_model)
        
        self.eemd = EEMD(s2e_pe_model=str(Path(base_url) / "s2e_ast_onto"))
        self.pemd = PEMD()

        self.preprocess = pe_data.PreProcess()
        self.postprocess = pe_data.PostProcess()

        # These are always initialize when get_annotations() is called
        self.conv_hist_for_pe = (
            []
        )  # initialize the history of conversation, which is used in PE Linking
        self.ment2ent = {}  # This will be used for PE Linking

    def _default_ed_model(self):
        from REL.entity_disambiguation import EntityDisambiguation
        return EntityDisambiguation(self.base_url, self.wiki_version, user_config={
                "mode": "eval",
                "model_path": f"{self.base_url}/{self.wiki_version}/generated/model",
            })

    def _error_check(self, conv):
        assert type(conv) == list
        for turn in conv:
            assert type(turn) == dict
            assert set(turn.keys()) == {"speaker", "utterance"}
            assert turn["speaker"] in [
                "USER",
                "SYSTEM",
            ], f'Speaker should be either "USER" or "SYSTEM", but got {turn["speaker"]}'

    def _el(self, utt):
        """Perform entity linking"""
        # MD
        md_results = self.bert_md.md(utt)

        # ED
        spans = [[r[0], r[1]] for r in md_results]  # r[0]: start, r[1]: length
        el_results = self.ed(utt, spans)  # ED

        self.conv_hist_for_pe[-1]["mentions"] = [r[2] for r in el_results]
        self.ment2ent.update(
            {r[2]: r[3] for r in el_results}
        )  # If there is a mismatch of annotations for the same mentions, the last one (the most closest turn's one to the PEM) will be used.

        return [r[:4] for r in el_results]  # [start_pos, length, mention, entity]

    def _pe(self, utt):
        """Perform PE Linking"""

        ret = []

        # Step 1: PE Mention Detection
        pem_results = self.pemd.pem_detector(utt)
        pem2result = {r[2]: r for r in pem_results}

        # Step 2: Finding corresponding explicit entity mentions (EEMs)
        # NOTE: Current implementation can handle only one target PEM at a time
        outputs = []
        for _, _, pem in pem_results:  # pems: [[start_pos, length, pem], ...]
            self.conv_hist_for_pe[-1]["pems"] = [
                pem
            ]  # Create a conv for each target PEM that you want to link

            # Preprocessing
            token_with_info = self.preprocess.get_tokens_with_info(
                self.conv_hist_for_pe
            )
            input_data = self.preprocess.get_input_of_pe_linking(token_with_info)

            assert (
                len(input_data) == 1
            ), f"Current implementation can handle only one target PEM at a time"
            input_data = input_data[0]

            # Finding corresponding explicit entity mentions (EEMs)
            scores = self.eemd.get_scores(input_data)

            # Post processing
            outputs += self.postprocess.get_results(
                input_data, self.conv_hist_for_pe, self.threshold, scores
            )

        self.conv_hist_for_pe[-1]["pems"] = []  # Remove the target PEM

        # Step 3: Get corresponding entity
        for r in outputs:
            pem = r["personal_entity_mention"]
            pem_result = pem2result[pem]  # [start_pos, length, pem]
            eem = r["mention"]  # Explicit entity mention
            ent = self.ment2ent[eem]  # Corresponding entity
            ret.append(
                [pem_result[0], pem_result[1], pem_result[2], ent]
            )  # [start_pos, length, PEM, entity]

        return ret

    def annotate(self, conv):
        """Get conversational entity linking annotations

        Args:
            conv: A list of dicts, each dict contains "speaker" and "utterance" keys.

        Returns:
            A list of dicts, each dict contains conv's ones + "annotations" key.
        """
        self._error_check(conv)
        ret = []
        self.conv_hist_for_pe = []  # Initialize
        self.ment2ent = {}  # Initialize

        for turn in conv:
            utt = turn["utterance"]
            assert turn["speaker"] in [
                "USER",
                "SYSTEM",
            ], f'Speaker should be either "USER" or "SYSTEM", but got {turn["speaker"]}'
            ret.append({"speaker": turn["speaker"], "utterance": utt})

            self.conv_hist_for_pe.append({})
            self.conv_hist_for_pe[-1]["speaker"] = turn["speaker"]
            self.conv_hist_for_pe[-1]["utterance"] = utt

            if turn["speaker"] == "USER":
                el_results = self._el(utt)
                pe_results = self._pe(utt)
                ret[-1]["annotations"] = el_results + pe_results

        return ret

    def ed(self, text, spans):
        """Change tuple to list to match the output format of REL API."""
        response = self.response_model.generate_response(text=text, spans=spans)
        return [list(ent) for ent in response]


