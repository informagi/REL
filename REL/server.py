from http.server import BaseHTTPRequestHandler
import torch
import time
import json

from REL.mention_detection import MentionDetection
from REL.utils import process_results
from flair.models import SequenceTagger

GERBIL = "gerbil_doc"


"""
Class/function combination that is used to setup an API that can be used for e.g. GERBIL evaluation.
"""


def make_handler(
    base_url, wiki_subfolder, model, tagger_ner, mode="ED", include_conf=False
):
    class GetHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.model = model
            self.tagger_ner = tagger_ner

            self.mode = mode
            self.include_conf = include_conf
            self.base_url = base_url
            self.wiki_subfolder = wiki_subfolder

            self.custom_ner = not isinstance(tagger_ner, SequenceTagger)
            self.use_offset = False if ((mode == "ED") or self.custom_ner) else True

            self.doc_cnt = 0
            self.mention_detection = MentionDetection(base_url, wiki_subfolder)

            super().__init__(*args, **kwargs)

        def do_POST(self):
            """
            Returns response.

            :return:
            """
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            self.send_response(200)
            self.end_headers()

            text, spans = self.read_json(post_data)
            response = self.generate_response(text, spans)

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
            spans = [(int(j["start"]), int(j["length"])) for j in data["spans"]]
            return text, spans

        def generate_response(self, text, spans):
            """
            Generates response for API. Can be either ED only or EL, meaning end-to-end.

            :return: list of tuples for each entity found.
            """

            n_words = len(text.split())

            start = time.time()
            if (self.mode == "ED") or self.custom_ner:
                if self.custom_ner:
                    spans = self.tagger_ner(text)

                # Verify if we have spans.
                if len(spans) == 0:
                    print("No spans found while in ED mode..?")
                    return []
                processed = {GERBIL: [text, spans]}  # self.split_text(text, spans)
                mentions_dataset, total_ment = self.mention_detection.format_spans(
                    processed
                )
            elif self.mode == "EL":
                # EL
                processed = {GERBIL: [text, spans]}
                mentions_dataset, total_ment = self.mention_detection.find_mentions(
                    processed, self.tagger_ner
                )
            else:
                raise Exception("Faulty mode, only valid options are: ED or EL")
            time_md = time.time() - start

            # Disambiguation
            start = time.time()
            predictions, timing = self.model.predict(mentions_dataset)
            time_ed = time.time() - start

            # Tuple of.
            efficiency = [str(n_words), str(total_ment), str(time_md), str(time_ed)]

            # write to txt file.
            with open('{}/{}/generated/efficiency.txt'.format(self.base_url, self.wiki_subfolder), 'a',
                      encoding='utf-8') as f:
                f.write('\t'.join(efficiency) + '\n')

            # Process result.
            result = process_results(
                mentions_dataset,
                predictions,
                processed,
                include_offset=self.use_offset,
                include_conf=self.include_conf,
            )

            result = result[GERBIL]
            self.doc_cnt += 1
            return result

    return GetHandler
