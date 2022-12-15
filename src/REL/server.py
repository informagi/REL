import json
from http.server import BaseHTTPRequestHandler

from flair.models import SequenceTagger

from REL.mention_detection import MentionDetection
from REL.utils import process_results

API_DOC = "API_DOC"



def make_handler(base_url, wiki_version, ed_model, tagger_ner, use_bert, process_sentences, split_docs_value=0):
    """
    Class/function combination that is used to setup an API that can be used for e.g. GERBIL evaluation.
    """
    class GetHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.ed_model = ed_model
            self.tagger_ner = tagger_ner
            self.use_bert = use_bert
            self.process_sentences = process_sentences
            self.split_docs_value = split_docs_value

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

        def do_HEAD(self):
            # send bad request response code
            self.send_response(400)
            self.end_headers()
            self.wfile.write(bytes(json.dumps([]), "utf-8"))
            return

        def do_POST(self):
            """
            Returns response.

            :return:
            """
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                self.send_response(200)
                self.end_headers()

                text, spans = self.read_json(post_data)
                response = self.generate_response(text, spans)

                self.wfile.write(bytes(json.dumps(response), "utf-8"))
            except Exception as e:
                print(f"Encountered exception: {repr(e)}")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(bytes(json.dumps([]), "utf-8"))
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
            if "spans" in data:
                try:
                    spans = [list(d.values()) for d in data["spans"]]
                except Exception:
                    spans = data["spans"]
                    pass
            else:
                spans = []

            return text, spans

        def convert_bert_result(self, result):
            new_result = {}
            for doc_key in result:
                new_result[doc_key] = []
                for mention_data in result[doc_key]:
                    new_result[doc_key].append(list(mention_data))
                    new_result[doc_key][-1][2], new_result[doc_key][-1][3] =\
                        new_result[doc_key][-1][3], new_result[doc_key][-1][2]
                    new_result[doc_key][-1] = tuple(new_result[doc_key][-1])
            return new_result

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
                    processed, self.use_bert, self.process_sentences, self.split_docs_value, self.tagger_ner
                )

            # Disambiguation
            predictions, timing = self.ed_model.predict(mentions_dataset)

            # Process result.
            result = process_results(
                mentions_dataset,
                predictions,
                processed,
                include_offset=False if ((len(spans) > 0) or self.custom_ner) else True,
            )
            # result = self.convert_bert_result(result)

            # Singular document.
            if len(result) > 0:
                return [*result.values()][0]

            return []

    return GetHandler


if __name__ == "__main__":
    import argparse
    from http.server import HTTPServer

    from REL.entity_disambiguation import EntityDisambiguation
    from REL.ner import load_flair_ner
    from REL.ner.bert_wrapper import load_bert_ner

    p = argparse.ArgumentParser()
    p.add_argument("base_url")
    p.add_argument("wiki_version")
    p.add_argument("--ed-model", default="ed-wiki-2019")
    p.add_argument("--ner-model", default="ner-fast")
    p.add_argument("--bind", "-b", metavar="ADDRESS", default="0.0.0.0")
    p.add_argument("--port", "-p", default=5555, type=int)
    p.add_argument("--use_bert_large_cased", help = "use Bert large cased rather than Flair", action="store_true")
    p.add_argument("--use_bert_base_cased", help = "use Bert base cased rather than Flair", action="store_true")
    p.add_argument("--use_bert_large_uncased", help = "use Bert large uncased rather than Flair", action="store_true")
    p.add_argument("--use_bert_base_uncased", help = "use Bert base uncased rather than Flair", action="store_true")
    p.add_argument("--process_sentences", help = "process sentences rather than documents", action="store_true")
    p.add_argument("--split_docs_value", help = "threshold number of tokens to split document")

    args = p.parse_args()

    use_bert_base_cased = False
    use_bert_large_cased = False
    use_bert_base_uncased = False
    use_bert_large_uncased = False

    if args.use_bert_base_cased:
        ner_model = load_bert_ner("dslim/bert-base-NER")
        use_bert_base_cased = True
    elif args.use_bert_large_cased:
        ner_model = load_bert_ner("dslim/bert-large-NER")
        use_bert_large_cased = True
    elif args.use_bert_base_uncased:
        ner_model = load_bert_ner("dslim/bert-base-NER-uncased")
        use_bert_base_uncased = True
    elif args.use_bert_large_uncased:
        ner_model = load_bert_ner("Jorgeutd/bert-large-uncased-finetuned-ner")
        use_bert_large_uncased = True
    else:
        ner_model = load_flair_ner(args.ner_model)

    split_docs_value = 0
    if args.split_docs_value:
        split_docs_value = int(args.split_docs_value)

    process_sentences = args.process_sentences

    ed_model = EntityDisambiguation(
        args.base_url, args.wiki_version, {"mode": "eval", "model_path": args.ed_model}
    )
    server_address = (args.bind, args.port)
    server = HTTPServer(
        server_address,
        make_handler(args.base_url, 
                     args.wiki_version, 
                     ed_model, 
                     ner_model, 
                     (use_bert_base_cased or use_bert_large_cased or use_bert_base_uncased or use_bert_large_uncased), 
                     process_sentences,
                     split_docs_value)
    )

    try:
        print("Ready for listening.")
        server.serve_forever()
    except KeyboardInterrupt:
        exit(0)
