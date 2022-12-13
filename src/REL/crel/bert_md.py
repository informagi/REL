import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


class BERT_MD:
    def __init__(self, file_pretrained):
        """

        Args:
            file_pretrained = "./tmp/ft-conel/"

        Note:
            The output of self.ner_model(s_input) is like
              - s_input: e.g, 'Burger King franchise'
              - return: e.g., [{'entity': 'B-ment', 'score': 0.99364895, 'index': 1, 'word': 'Burger', 'start': 0, 'end': 6}, ...]
        """

        model = AutoModelForTokenClassification.from_pretrained(file_pretrained)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(file_pretrained)
        self.ner_model = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=device.index if device.index != None else -1,
            ignore_labels=[],
        )

    def md(self, s, flag_warning=False):
        """Perform mention detection

        Args:
            s: input string
            flag_warning: if True, print warning message

        Returns:
            REL style annotation results: [(start_position, length, mention), ...]
            E.g., [[0, 15, 'The Netherlands'], ...]
        """

        ann = self.ner_model(s)  # Get ann results from BERT-NER model

        ret = []
        pos_start, pos_end = -1, -1  # Initialize variables

        for i in range(len(ann)):
            w, ner = ann[i]["word"], ann[i]["entity"]
            assert ner in [
                "B-ment",
                "I-ment",
                "O",
            ], f"Unexpected ner tag: {ner}. If you use BERT-NER as it is, then you should flag_use_normal_bert_ner_tag=True."
            if ner == "B-ment" and w[:2] != "##":
                if (pos_start != -1) and (pos_end != -1):  # If B-ment is already found
                    ret.append(
                        [pos_start, pos_end - pos_start, s[pos_start:pos_end]]
                    )  # save the previously identified mention
                    pos_start, pos_end = -1, -1  # Initialize
                pos_start, pos_end = ann[i]["start"], ann[i]["end"]

            elif ner == "B-ment" and w[:2] == "##":
                if (ann[i]["index"] == ann[i - 1]["index"] + 1) and (
                    ann[i - 1]["entity"] != "B-ment"
                ):  # If previous token has an entity (ner) label and it is NOT "B-ment" (i.e., ##xxx should not be the begin of the entity)
                    if flag_warning:
                        print(
                            f"WARNING: ##xxx (in this case {w}) should not be the begin of the entity"
                        )

            elif (
                i > 0
                and (ner == "I-ment")
                and (ann[i]["index"] == ann[i - 1]["index"] + 1)
            ):  # If w is I-ment and previous word's index (i.e., ann[i-1]['index']) is also a mention
                pos_end = ann[i]["end"]  # update pos_end

            # This only happens when flag_ignore_o is False
            elif (
                ner == "O"
                and w[:2] == "##"
                and (
                    ann[i - 1]["entity"] == "B-ment" or ann[i - 1]["entity"] == "I-ment"
                )
            ):  # If w is "O" and ##xxx, and previous token's index (i.e., ann[i-1]['index']) is B-ment or I-ment
                pos_end = ann[i]["end"]  # update pos_end

        # Append remaining ment
        if (pos_start != -1) and (pos_end != -1):
            ret.append(
                [pos_start, pos_end - pos_start, s[pos_start:pos_end]]
            )  # Save last mention

        return ret
