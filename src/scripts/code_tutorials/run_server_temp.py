from http.server import HTTPServer

# --------------------- Overwrite class
from typing import Dict

import flair
import torch
import torch.nn
from flair.data import Dictionary as DDD
from flair.embeddings import TokenEmbeddings
from flair.models import SequenceTagger
from torch.nn.parameter import Parameter

from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import load_flair_ner
from REL.server import make_handler


def _init_initial_hidden_state(self, num_directions: int):
    hs_initializer = torch.nn.init.xavier_normal_
    lstm_init_h = torch.nn.Parameter(
        torch.zeros(self.rnn.num_layers * num_directions, self.hidden_size),
        requires_grad=True,
    )
    lstm_init_c = torch.nn.Parameter(
        torch.zeros(self.rnn.num_layers * num_directions, self.hidden_size),
        requires_grad=True,
    )
    return hs_initializer, lstm_init_h, lstm_init_c


SequenceTagger._init_initial_hidden_state = _init_initial_hidden_state
# ---------------------


def user_func(text):
    spans = [(0, 5), (17, 7), (50, 6)]
    return spans


# 0. Set your project url, which is used as a reference for your datasets etc.
base_url = "/store/projects/REL"
wiki_version = "wiki_2019"

# 1. Init model, where user can set his/her own config that will overwrite the default config.
# If mode is equal to 'eval', then the model_path should point to an existing model.
config = {
    "mode": "eval",
    "model_path": "{}/{}/generated/model".format(base_url, wiki_version),
}

model = EntityDisambiguation(base_url, wiki_version, config)

# 2. Create NER-tagger.
tagger_ner = load_flair_ner("ner-fast-with-lowercase")

# 2.1. Alternatively, one can create his/her own NER-tagger that given a text,
# returns a list with spans (start_pos, length).
# tagger_ner = user_func

# 3. Init server.
server_address = ("127.0.0.1", 1235)
server = HTTPServer(
    server_address,
    make_handler(base_url, wiki_version, model, tagger_ner),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)
