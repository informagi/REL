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

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


def __init__(
    self,
    hidden_size: int,
    embeddings: TokenEmbeddings,
    tag_dictionary: DDD,
    tag_type: str,
    use_crf: bool = True,
    use_rnn: bool = True,
    rnn_layers: int = 1,
    dropout: float = 0.0,
    word_dropout: float = 0.05,
    locked_dropout: float = 0.5,
    train_initial_hidden_state: bool = False,
    rnn_type: str = "LSTM",
    pickle_module: str = "pickle",
    beta: float = 1.0,
    loss_weights: Dict[str, float] = None,
):
    """
    Initializes a SequenceTagger
    :param hidden_size: number of hidden states in RNN
    :param embeddings: word embeddings used in tagger
    :param tag_dictionary: dictionary of tags you want to predict
    :param tag_type: string identifier for tag type
    :param use_crf: if True use CRF decoder, else project directly to tag space
    :param use_rnn: if True use RNN layer, otherwise use word embeddings directly
    :param rnn_layers: number of RNN layers
    :param dropout: dropout probability
    :param word_dropout: word dropout probability
    :param locked_dropout: locked dropout probability
    :param train_initial_hidden_state: if True, trains initial hidden state of RNN
    :param beta: Parameter for F-beta score for evaluation and training annealing
    :param loss_weights: Dictionary of weights for classes (tags) for the loss function
    (if any tag's weight is unspecified it will default to 1.0)
    """

    super(SequenceTagger, self).__init__()
    self.use_rnn = use_rnn
    self.hidden_size = hidden_size
    self.use_crf: bool = use_crf
    self.rnn_layers: int = rnn_layers

    self.trained_epochs: int = 0

    self.embeddings = embeddings

    # set the dictionaries
    self.tag_dictionary: DDD = tag_dictionary
    self.tag_type: str = tag_type
    self.tagset_size: int = len(tag_dictionary)

    self.beta = beta

    self.weight_dict = loss_weights
    # Initialize the weight tensor
    if loss_weights is not None:
        n_classes = len(self.tag_dictionary)
        weight_list = [1.0 for i in range(n_classes)]
        for i, tag in enumerate(self.tag_dictionary.get_items()):
            if tag in loss_weights.keys():
                weight_list[i] = loss_weights[tag]
        self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
    else:
        self.loss_weights = None

    # initialize the network architecture
    self.nlayers: int = rnn_layers
    self.hidden_word = None

    # dropouts
    self.use_dropout: float = dropout
    self.use_word_dropout: float = word_dropout
    self.use_locked_dropout: float = locked_dropout

    self.pickle_module = pickle_module

    if dropout > 0.0:
        self.dropout = torch.nn.Dropout(dropout)

    if word_dropout > 0.0:
        self.word_dropout = flair.nn.WordDropout(word_dropout)

    if locked_dropout > 0.0:
        self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

    rnn_input_dim: int = self.embeddings.embedding_length

    self.relearn_embeddings: bool = True

    if self.relearn_embeddings:
        self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

    self.train_initial_hidden_state = train_initial_hidden_state
    self.bidirectional = True
    self.rnn_type = rnn_type

    # bidirectional LSTM on top of embedding layer
    if self.use_rnn:
        num_directions = 2 if self.bidirectional else 1

        if self.rnn_type in ["LSTM", "GRU"]:

            self.rnn = getattr(torch.nn, self.rnn_type)(
                rnn_input_dim,
                hidden_size,
                num_layers=self.nlayers,
                dropout=0.0 if self.nlayers == 1 else 0.5,
                bidirectional=True,
                batch_first=True,
            )
            # Create initial hidden state and initialize it
            if self.train_initial_hidden_state:
                self.hs_initializer = torch.nn.init.xavier_normal_

                self.lstm_init_h = Parameter(
                    torch.zeros(
                        self.nlayers * num_directions, self.hidden_size
                    ).float(),
                    requires_grad=True,
                )

                self.lstm_init_c = Parameter(
                    torch.zeros(
                        self.nlayers * num_directions, self.hidden_size
                    ).float(),
                    requires_grad=True,
                )

                # TODO: Decide how to initialize the hidden state variables
                # self.hs_initializer(self.lstm_init_h)
                # self.hs_initializer(self.lstm_init_c)

        # final linear map to tag space
        self.linear = torch.nn.Linear(hidden_size * num_directions, len(tag_dictionary))
    else:
        self.linear = torch.nn.Linear(
            self.embeddings.embedding_length, len(tag_dictionary)
        )

    if self.use_crf:
        self.transitions = torch.nn.Parameter(
            torch.zeros(self.tagset_size, self.tagset_size).float()
        )

        self.transitions.detach()[
            self.tag_dictionary.get_idx_for_item(START_TAG), :
        ] = -10000

        self.transitions.detach()[
            :, self.tag_dictionary.get_idx_for_item(STOP_TAG)
        ] = -10000

    self.to(flair.device)


SequenceTagger.__init__ = __init__
# ---------------------


def user_func(text):
    spans = [(0, 5), (17, 7), (50, 6)]
    return spans


# 0. Set your project url, which is used as a reference for your datasets etc.
base_url = "C:/Users/mickv/Desktop/data_back/"
wiki_version = "wiki_2019"

# 1. Init model, where user can set his/her own config that will overwrite the default config.
# If mode is equal to 'eval', then the model_path should point to an existing model.
config = {
    "mode": "eval",
    "model_path": "{}/{}/generated/model".format(base_url, wiki_version),
}

model = EntityDisambiguation(base_url, wiki_version, config)

# 2. Create NER-tagger.
tagger_ner = load_flair_ner("ner-fast")

# 2.1. Alternatively, one can create his/her own NER-tagger that given a text,
# returns a list with spans (start_pos, length).
# tagger_ner = user_func

# 3. Init server.
server_address = ("192.168.178.11", 1235)
server = HTTPServer(
    server_address,
    make_handler(base_url, wiki_version, model, tagger_ner, include_conf=True),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)
