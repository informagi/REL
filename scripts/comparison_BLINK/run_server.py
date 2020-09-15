import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

import blink.main_dense as main_dense
from flair.models import SequenceTagger

from REL.mention_detection import MentionDetection
from REL.ner import load_flair_ner
from REL.utils import split_in_words

API_DOC = "API_DOC"


def process_results(
    mentions_dataset, predictions, processed, include_offset=False,
):
    """
    Function that can be used to process the End-to-End results.

    :return: dictionary with results and document as key.
    """

    res = {}
    for doc in mentions_dataset:
        if doc not in predictions:
            # No mentions found, we return empty list.
            continue
        pred_doc = predictions[doc]
        ment_doc = mentions_dataset[doc]
        text = processed[doc][0]

        res_doc = []

        for pred, ment in zip(pred_doc, ment_doc):
            sent = ment["sentence"]

            # Only adjust position if using Flair NER tagger.
            if include_offset:
                offset = text.find(sent)
            else:
                offset = 0
            start_pos = offset + ment["pos"]
            mention_length = int(ment["end_pos"] - ment["pos"])

            # self.verify_pos(ment["ngram"], start_pos, end_pos, text)
            if pred["prediction"] != "NIL":
                temp = (
                    start_pos,
                    mention_length,
                    pred["prediction"],
                    ment["ngram"],
                    ment["conf_md"] if "conf_md" in ment else -1,
                    ment["tag"] if "tag" in ment else "NULL",
                )
                res_doc.append(temp)
        res[doc] = res_doc
    return res


def _get_ctxt(self, start, end, idx_sent, sentence):
    """
    Retrieves context surrounding a given mention up to 100 words from both sides.

    :return: left and right context
    """

    # Iteratively add words up until we have 100
    left_ctxt = split_in_words(sentence[:start])
    left_ctxt = " ".join(left_ctxt)

    right_ctxt = split_in_words(sentence[end:])
    right_ctxt = " ".join(right_ctxt)

    return left_ctxt, right_ctxt


# Overwrite to just get current sentence context.
MentionDetection._get_ctxt = _get_ctxt


def make_handler(base_url, wiki_version, models, tagger_ner, argss, logger):
    class GetHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.model = models
            self.tagger_ner = tagger_ner

            self.argss = argss
            self.logger = logger

            self.base_url = base_url
            self.wiki_version = wiki_version

            self.custom_ner = not isinstance(tagger_ner, SequenceTagger)
            self.mention_detection = MentionDetection(base_url, wiki_version)

            super().__init__(*args, **kwargs)

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                bytes(
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "label": "status",
                            "message": "up",
                            "color": "green",
                        }
                    ),
                    "utf-8",
                )
            )
            return

        def do_POST(self):
            """
            Returns response.

            :return:
            """
            content_length = int(self.headers["Content-Length"])
            print(content_length)
            post_data = self.rfile.read(content_length)
            self.send_response(200)
            self.end_headers()

            text, spans = self.read_json(post_data)
            response = self.generate_response(text, spans)

            print(response)
            print("=========")

            # print('response in server.py code:\n\n {}'.format(response))
            self.wfile.write(bytes(json.dumps(response), "utf-8"))
            return

        def read_json(self, post_data):
            """
            Reads input JSON message.

            :return: document text and spans.
            """

            data = json.loads(post_data.decode("utf-8"))
            text = data["text"]
            text = text.replace("&amp;", "&")

            # GERBIL sends dictionary, users send list of lists.
            try:
                spans = [list(d.values()) for d in data["spans"]]
            except Exception:
                spans = data["spans"]
                pass

            return text, spans

        def generate_response(self, text, spans):
            """
            Generates response for API. Can be either ED only or EL, meaning end-to-end.

            :return: list of tuples for each entity found.
            """

            if len(text) == 0:
                return []

            if len(spans) > 0:
                # ED.
                processed = {API_DOC: [text, spans]}
                mentions_dataset, total_ment = self.mention_detection.format_spans(
                    processed
                )
            else:
                # EL
                processed = {API_DOC: [text, spans]}
                mentions_dataset, total_ment = self.mention_detection.find_mentions(
                    processed, self.tagger_ner
                )

            # Create to-be-linked dataset.
            data_to_link = []
            temp_m = mentions_dataset[API_DOC]
            for i, m in enumerate(temp_m):
                # Using ngram, which is basically the original mention (without preprocessing as in BLINK's code).
                temp = {
                    "id": i,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": m["context"][0].lower(),
                    "mention": m["ngram"].lower(),
                    "context_right": m["context"][1].lower(),
                }
                data_to_link.append(temp)
            _, _, _, _, _, predictions, scores, = main_dense.run(
                self.argss, self.logger, *self.model, test_data=data_to_link
            )

            predictions = {
                API_DOC: [{"prediction": x[0].replace(" ", "_")} for x in predictions]
            }
            # Process result.
            result = process_results(
                mentions_dataset,
                predictions,
                processed,
                include_offset=False if ((len(spans) > 0) or self.custom_ner) else True,
            )

            # Singular document.
            if len(result) > 0:
                return [*result.values()][0]

            return []

    return GetHandler


# --------------

# 0. Set your project url, which is used as a reference for your datasets etc.
base_url = "/users/vanhulsm/Desktop/projects/data/"
wiki_version = "wiki_2014"

# 1. Init model, where user can set his/her own config that will overwrite the default config.
# If mode is equal to 'eval', then the model_path should point to an existing model.
config = {
    "mode": "eval",
    "model_path": "{}/{}/generated/model".format(base_url, wiki_version),
}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models_path = "/users/vanhulsm/Desktop/projects/BLINK/models/"  # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "biencoder_model": models_path + "biencoder_wiki_large.bin",
    "biencoder_config": models_path + "biencoder_wiki_large.json",
    "entity_catalogue": models_path + "entity.jsonl",
    "entity_encoding": models_path + "all_entities_large.t7",
    "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
    "crossencoder_config": models_path + "crossencoder_wiki_large.json",
    "fast": True,  # set this to be true if speed is a concern
    "output_path": "logs/",  # logging directory
    "top_k": 1,
}

args = argparse.Namespace(**config)
models = main_dense.load_models(args)


# 2. Create NER-tagger.
tagger_ner = load_flair_ner("ner-fast")

# 3. Init server.
server_address = ("localhost", 5555)
server = HTTPServer(
    server_address,
    make_handler(base_url, wiki_version, models, tagger_ner, args, logger),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)
