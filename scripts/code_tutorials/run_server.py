import argparse
from http.server import HTTPServer

from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import load_flair_ner
from REL.server import make_handler

p = argparse.ArgumentParser()
p.add_argument("base_url")
p.add_argument("wiki_version")
p.add_argument("--ed-model", default="ed-wiki-2019")
p.add_argument("--ner-tagger", default="ner-fast")
p.add_argument("--port", type=int, default=5555)
args = p.parse_args()

# Set some arguments
base_url = args.base_url
wiki_version = args.wiki_version
port = args.port
ed_model = args.ed_model
ner_tagger = args.ner_tagger

# Init model, where user can set his/her own config that will overwrite the default
# config.  If mode is equal to 'eval', then the model_path should point to an existing
# model (alias/path/URL).
config = {
    "mode": "eval",
    "model_path": ed_model,
}

model = EntityDisambiguation(base_url, wiki_version, config)

# 2. Create NER-tagger.
ner_tagger = load_flair_ner(ner_tagger)  # or another tagger

# 3. Init server.
server_address = ("0.0.0.0", port)
server = HTTPServer(
    server_address, make_handler(base_url, wiki_version, model, ner_tagger),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)
