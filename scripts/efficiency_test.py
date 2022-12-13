import numpy as np
import requests
import os

from REL.training_datasets import TrainingEvaluationDatasets

np.random.seed(seed=42)

base_url = os.environ.get("REL_BASE_URL")
wiki_version = "wiki_2019"
host = 'localhost'
port = '5555'
datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()["aida_testB"]

# random_docs = np.random.choice(list(datasets.keys()), 50)

server = True
docs = {}
for i, doc in enumerate(datasets):
    sentences = []
    for x in datasets[doc]:
        if x["sentence"] not in sentences:
            sentences.append(x["sentence"])
    text = ". ".join([x for x in sentences])

    if len(docs) == 50:
        print("length docs is 50.")
        print("====================")
        break

    if len(text.split()) > 200:
        docs[doc] = [text, []]
        # Demo script that can be used to query the API.
        if server:
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
            print(requests.post(f"http://{host}:{port}", json=myjson).json())
            print("----------------------------")


# --------------------- Now total --------------------------------
# ------------- RUN SEPARATELY TO BALANCE LOAD--------------------
if not server:
    from time import time

    import flair
    import torch
    from flair.models import SequenceTagger

    from REL.entity_disambiguation import EntityDisambiguation
    from REL.mention_detection import MentionDetection

    base_url = "C:/Users/mickv/desktop/data_back/"

    flair.device = torch.device("cuda:0")

    mention_detection = MentionDetection(base_url, wiki_version)

    # Alternatively use Flair NER tagger.
    tagger_ner = SequenceTagger.load("ner-fast")

    start = time()
    mentions_dataset, n_mentions = mention_detection.find_mentions(docs, tagger_ner)
    print("MD took: {}".format(time() - start))

    # 3. Load model.
    config = {
        "mode": "eval",
        "model_path": "{}/{}/generated/model".format(base_url, wiki_version),
    }
    model = EntityDisambiguation(base_url, wiki_version, config)

    # 4. Entity disambiguation.
    start = time()
    predictions, timing = model.predict(mentions_dataset)
    print("ED took: {}".format(time() - start))
