from http.server import HTTPServer

from flair.models import SequenceTagger

from REL.entity_disambiguation import EntityDisambiguation
from REL.server import make_handler
from REL.ngram import Cmns
from REL.example_custom_MD import MD_Module

# 0. Set your project url, which is used as a reference for your datasets etc.
base_url = "/users/vanhulsm/Desktop/projects/data/"
wiki_version = "wiki_2014"

# 1. Init model, where user can set his/her own config that will overwrite the default config.
# If mode is equal to 'eval', then the model_path should point to an existing model.
config = {
    "mode": "eval",
    "model_path": "{}/{}/generated/model".format(
        base_url, wiki_version
    ),
}

model = EntityDisambiguation(base_url, wiki_version, config)

# 2. Create NER-tagger.
tagger_ner = SequenceTagger.load("ner-fast")
tagger_custom = MD_Module('param1', 'param2')
tagger_ngram = Cmns(base_url, wiki_version, n=5)

# 3. Init server.
server_address = ("127.0.0.1", 1235)
server = HTTPServer(
    server_address,
    make_handler(
        base_url, wiki_version, model, tagger_ngram
    ),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)
