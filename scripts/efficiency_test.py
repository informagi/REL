import argparse
from scripts.evaluate_predictions import evaluate
import json
import numpy as np
import os
import re
import requests

from REL.ner.set_tagger_ner import set_tagger_ner
from REL.training_datasets import TrainingEvaluationDatasets

parser = argparse.ArgumentParser()
parser.add_argument("--max_docs", help = "number of documents")
parser.add_argument("--process_sentences", help = "process sentences rather than documents", action="store_true")
parser.add_argument("--split_docs_value", help = "threshold number of tokens to split document")
parser.add_argument("--tagger_ner_name", help = "mention detection tagger")
parser.add_argument("--use_server", help = "use server", action="store_true")
parser.add_argument("--wiki_version", help = "Wiki version")
args = parser.parse_args()

np.random.seed(seed=42)

base_url = os.path.abspath(os.path.dirname(__file__) + "/../data/")
process_sentences = args.process_sentences

if args.split_docs_value:
    split_docs_value = int(args.split_docs_value)
else:
    split_docs_value = 0

if args.max_docs:
    max_docs = int(args.max_docs)
else:
    max_docs = 50

if args.wiki_version:
    wiki_version = args.wiki_version
else:
    wiki_version = "wiki_2019"

datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()["aida_testB"]

use_server = args.use_server

docs = {}
all_results = {}
for i, doc in enumerate(datasets):
    sentences = []
    for x in datasets[doc]:
        if x["sentence"] not in sentences:
            sentences.append(x["sentence"])
    if len(sentences) == 0:
        next

    text = ". ".join([x for x in sentences])
    if len(docs) >= max_docs:
        print(f"length docs is {len(docs)}.")
        print("====================")
        break

    if len(text.split()) > 200:
        docs[doc] = [text, []]
        # Demo script that can be used to query the API.
        if use_server:
            print(f"max_docs={max_docs} use_server={use_server}")
            myjson = {
                "text": text,
                "spans": [
                    # {"start": 41, "length": 16}
                ],
            }
            print("----------------------------")
            print(i, "Input API:")
            print(myjson)

            print("Output API:")
            results = requests.post("http://0.0.0.0:5555", json=myjson)
            print(results.json())
            print("----------------------------")
            try:
                results_list = []
                for result in results.json():
                    results_list.append({ "mention": result[2], "prediction": result[3] }) # Flair + Bert
                all_results[doc] = results_list
            except json.decoder.JSONDecodeError:
                print("The analysis results are not in json format:", str(results))
                all_results[doc] = []

if len(all_results) > 0:
    evaluate(all_results)

# --------------------- Now total --------------------------------
# ------------- RUN SEPARATELY TO BALANCE LOAD--------------------
if not use_server:
    from time import time

    import flair
    import torch
    from flair.models import SequenceTagger

    from REL.entity_disambiguation import EntityDisambiguation
    from REL.mention_detection import MentionDetection

    from REL.ner.bert_wrapper import load_bert_ner

    flair.device = torch.device("cpu")
    tagger_ner_name = args.tagger_ner_name
    tagger_ner = set_tagger_ner(tagger_ner_name)

    print(f"max_docs={max_docs} tagger_ner_name={tagger_ner_name} wiki_version={wiki_version} process_sentences={process_sentences} split_docs_value={split_docs_value}")

    mention_detection = MentionDetection(base_url, wiki_version)

    start = time()
    mentions_dataset, n_mentions = mention_detection.find_mentions(docs, tagger_ner_name, process_sentences, split_docs_value, tagger_ner)
    print("MD took: {} seconds".format(round(time() - start, 2)))

    # 3. Load ED model.
    config = {
        "mode": "eval",
        "model_path": "{}/{}/generated/model".format(base_url, wiki_version),
    }
    ed_model = EntityDisambiguation(base_url, wiki_version, config)

    # 4. Entity disambiguation.
    start = time()
    predictions, timing = ed_model.predict(mentions_dataset)
    print("ED took: {} seconds".format(round(time() - start, 2)))

    evaluate(predictions)
