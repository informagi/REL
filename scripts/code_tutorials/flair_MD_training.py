import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.adam import Adam

import flair
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.datasets import NCBI_DISEASE
from flair.models import SequenceTagger

'''
TODO:
1. Replace final few layers Flair and add Sigmoid (true/false). <-- not sure if we should use the pretrained model
from NER or the POS-tagger? I'd probably say POS-tagger.
2. Create dataset from AIDA (input should be random combinations) <-- see E2E paper.
3. This gives us a binary dataset to find mentions.
4. If works well, can also try E2E OR use confidence score MD as input
to ED model. OR combine both output scores..? p(x,y) = p(x)*p(y)


https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md

'''

# 4. create the text classifier (else train from scratch).
train_scratch = False
if train_scratch:
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)
else:
    #TODO: if I can somehow replace the final layers, then I can fine-tune it using our new dataset.
    classifier = SequenceTagger.load("pos-fast")
    print(classifier.linear)


from typing import List, Union, Optional, Dict
from flair.data import Sentence

def forward(self, sentences: List[Sentence]):
    self.embeddings.embed(sentences)

    names = self.embeddings.get_names()

    lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
    longest_token_sequence_in_batch: int = max(lengths)

    pre_allocated_zero_tensor = torch.zeros(
        self.embeddings.embedding_length * longest_token_sequence_in_batch,
        dtype=torch.float,
        device=flair.device,
    )

    all_embs = list()
    for sentence in sentences:
        all_embs += [
            emb for token in sentence for emb in token.get_each_embedding(names)
        ]
        nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

        if nb_padding_tokens > 0:
            t = pre_allocated_zero_tensor[
                : self.embeddings.embedding_length * nb_padding_tokens
                ]
            all_embs.append(t)

    sentence_tensor = torch.cat(all_embs).view(
        [
            len(sentences),
            longest_token_sequence_in_batch,
            self.embeddings.embedding_length,
        ]
    )

    # --------------------------------------------------------------------
    # FF PART
    # --------------------------------------------------------------------
    if self.use_dropout > 0.0:
        sentence_tensor = self.dropout(sentence_tensor)
    if self.use_word_dropout > 0.0:
        sentence_tensor = self.word_dropout(sentence_tensor)
    if self.use_locked_dropout > 0.0:
        sentence_tensor = self.locked_dropout(sentence_tensor)

    if self.reproject_embeddings:
        sentence_tensor = self.embedding2nn(sentence_tensor)

    if self.use_rnn:
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            sentence_tensor, lengths, enforce_sorted=False, batch_first=True
        )

        # if initial hidden state is trainable, use this state
        if self.train_initial_hidden_state:
            initial_hidden_state = [
                self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
                self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
            ]
            rnn_output, hidden = self.rnn(packed, initial_hidden_state)
        else:
            rnn_output, hidden = self.rnn(packed)

        sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            rnn_output, batch_first=True
        )

        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        # word dropout only before LSTM - TODO: more experimentation needed
        # if self.use_word_dropout > 0.0:
        #     sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

    if self.custom_layers_MD:
        x = self.drop1(F.relu(self.fc1(sentence_tensor)))
        x = self.drop2(F.relu(self.fc2(x)))
        features = self.linear(x)
    else:
        features = self.linear(sentence_tensor)
    return features

# Set custom layers
classifier.custom_layers_MD = True

classifier.fc1 = nn.Linear(512, 256)
classifier.drop1 = nn.Dropout(0.3)

classifier.fc2 = nn.Linear(256, 256)
classifier.drop2= nn.Dropout(0.3)

#TODO: Think the trainer object uses softmax for binary, but need to check.
classifier.linear = nn.Linear(256, 2)

# Monkey patch
classifier.forward = forward.__get__(classifier, SequenceTagger)

from REL.entity_disambiguation import EntityDisambiguation
from REL.training_datasets import TrainingEvaluationDatasets

base_url = "/users/vanhulsm/Desktop/projects/data"
wiki_version = "wiki_2019"

# 1. Load datasets # '/mnt/c/Users/mickv/Google Drive/projects/entity_tagging/deep-ed/data/wiki_2019/'
datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()

# 1. get the corpus
# TODO: Look into how the formatting is done, then replicate with AIDA.
# TODO: Replace with AIDA-A
corpus: Corpus = TREC_6()

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# # 3. initialize transformer document embeddings (many models are available).
# document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True)

# 5. initialize the text classifier trainer with Adam optimizer
trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

# 6. start the training
# 9. continue trainer at later point
from pathlib import Path

# checkpoint = 'resources/taggers/example-ner/checkpoint.pt'
# trainer = ModelTrainer.load_checkpoint(checkpoint, corpus)
trainer.train('resources/taggers/trec',
              learning_rate=3e-5, # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
              max_epochs=5, # terminate after 5 epochs
              checkpoint=True)