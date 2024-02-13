from http.server import HTTPServer

from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import load_flair_ner
from REL.server import make_handler

# 0. Set your project url, which is used as a reference for your datasets etc.
base_url = ""
wiki_version = "wiki_2019"

# 1. Init model, where user can set his/her own config that will overwrite the default config.
# If mode is equal to 'eval', then the model_path should point to an existing model.
config = {
    "mode": "eval",
    "model_path": "{}/{}/generated/model".format(base_url, wiki_version),
}

model = EntityDisambiguation(base_url, wiki_version, config)

# 2. Create NER-tagger.
tagger_ner = load_flair_ner("ner-fast")  # or another tagger

# 3. Init server.
server_address = ("127.0.0.1", 5555)
server = HTTPServer(
    server_address,
    make_handler(base_url, wiki_version, model, tagger_ner),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)
