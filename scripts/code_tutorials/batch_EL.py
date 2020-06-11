from REL.training_datasets import TrainingEvaluationDatasets
import numpy as np
import flair
import torch

from flair.models import SequenceTagger
from REL.mention_detection import MentionDetection
from time import time

np.random.seed(seed=42)

MAX_SIZE_DOCS = 10
base_url = ""
wiki_version = ""
datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()["aida_testB"]

docs = {}
for i, doc in enumerate(datasets):
    sentences = []
    for x in datasets[doc]:
        if x["sentence"] not in sentences:
            sentences.append(x["sentence"])
    text = ". ".join([x for x in sentences])

    if len(docs) == MAX_SIZE_DOCS:
        print("length docs is {}.".format(len(docs)))
        print("====================")
        break

    if len(text.split()) > 200:
        docs[doc] = [text, []]

mention_detection = MentionDetection(base_url, wiki_version)

# Alternatively use Flair NER tagger.
tagger_ner = SequenceTagger.load("ner-fast")

start = time()
mentions_dataset, n_mentions = mention_detection.find_mentions(docs, tagger_ner)
print("MD took: {}".format(time() - start))
