from http.server import HTTPServer

import flair
import torch

# from flair.models import SequenceTagger
from REL.seq_tagger_temp import SequenceTagger

from REL.entity_disambiguation import EntityDisambiguation
from REL.server import make_handler

def user_func(text):
    spans = [(0, 5), (17, 7), (50, 6)]
    return spans


# 0. Set your project url, which is used as a reference for your datasets etc.
base_url = "/mnt/c/Users/mickv/Desktop/data_back/"
wiki_subfolder = "wiki_2019"

# 1. Init model, where user can set his/her own config that will overwrite the default config.
# If mode is equal to 'eval', then the model_path should point to an existing model.
config = {
    "mode": "eval",
    "model_path": "{}/{}/generated/model".format(
        base_url, wiki_subfolder
    ),
}

print('AAAAA')
model = EntityDisambiguation(base_url, wiki_subfolder, config)

# 2. Create NER-tagger.
tagger_ner = SequenceTagger.load("ner-fast")

# 2.1. Alternatively, one can create his/her own NER-tagger that given a text,
# returns a list with spans (start_pos, length).
# tagger_ner = user_func

# 3. Init server.
MODE = "EL"
server_address = ("localhost", 5555)
server = HTTPServer(
    server_address,
    make_handler(
        base_url, wiki_subfolder, model, tagger_ner, mode=MODE, include_conf=True
    ),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)
