import argparse
import evaluate_predictions
import json
import numpy as np
import requests

from REL.training_datasets import TrainingEvaluationDatasets

parser = argparse.ArgumentParser()
parser.add_argument("--max_docs", help = "number of documents")
parser.add_argument("--process_sentences", help = "process sentences rather than documents", action="store_true")
parser.add_argument("--split_docs_value", help = "threshold number of tokens to split document")
parser.add_argument("--use_bert_base_cased", help = "use Bert base cased rather than Flair", action="store_true")
parser.add_argument("--use_bert_large_cased", help = "use Bert large cased rather than Flair", action="store_true")
parser.add_argument("--use_bert_base_uncased", help = "use Bert base uncased rather than Flair", action="store_true")
parser.add_argument("--use_bert_large_uncased", help = "use Bert large uncased rather than Flair", action="store_true")
parser.add_argument("--use_bert_multilingual", help = "use Bert multilingual rather than Flair", action="store_true")
parser.add_argument("--use_server", help = "use server", action="store_true")
parser.add_argument("--wiki_version", help = "Wiki version")
args = parser.parse_args()

np.random.seed(seed=42)

base_url = "/store/userdata/etjong/REL-erik/data/"
if args.max_docs:
    max_docs = int(args.max_docs)
else:
    max_docs = 50
if args.process_sentences:
    process_sentences = True
else:
    process_sentences = False

if args.split_docs_value:
    split_docs_value = int(args.split_docs_value)
else:
    split_docs_value = 0

if args.wiki_version:
    wiki_version = args.wiki_version
else:
    wiki_version = "wiki_2019"

datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()["aida_testB"]

if args.use_server:
    use_server = True
else:
    use_server = False

use_bert_base_cased = False
use_bert_large_cased = False
use_bert_base_uncased = False
use_bert_large_uncased = False
use_bert_multilingual = False

if args.use_bert_base_cased:
    use_bert_base_cased = True
elif args.use_bert_large_cased:
    use_bert_large_cased = True
elif args.use_bert_base_uncased:
    use_bert_base_uncased = True
elif args.use_bert_large_uncased:
    use_bert_large_uncased = True
elif args.use_bert_multilingual:
    use_bert_multilingual = True

print(f"max_docs={max_docs} wiki_version={wiki_version} use_bert_base_cased={use_bert_base_cased} use_bert_large_cased={use_bert_large_cased} use_bert_base_uncased={use_bert_base_uncased} use_bert_large_uncased={use_bert_large_uncased} use_bert_multilingual={use_bert_multilingual} use_server={use_server} process_sentences={process_sentences} split_docs_value={split_docs_value}")

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
            print("----------------------------")
            print("results", type(results), results)
            print("results.json()", results.json())
            try:
                results_list = []
                print("eff_test", results.json())
                for result in results.json():
                    results_list.append({ "mention": result[2], "prediction": result[3] }) # Flair + Bert
                all_results[doc] = results_list
            except json.decoder.JSONDecodeError:
                print("The analysis results are not in json format:", str(results))
                all_results[doc] = []

if len(all_results) > 0:
    evaluate_predictions.evaluate(all_results)

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

    mention_detection = MentionDetection(base_url, wiki_version)

    # Alternatively use Flair NER tagger.
    if use_bert_base_uncased:
        tagger_ner = load_bert_ner("dslim/bert-base-NER-uncased")
    elif use_bert_large_uncased:
        tagger_ner = load_bert_ner("Jorgeutd/bert-large-uncased-finetuned-ner")
    elif use_bert_base_cased:
        tagger_ner = load_bert_ner("dslim/bert-base-NER")
    elif use_bert_large_cased:
        tagger_ner = load_bert_ner("dslim/bert-large-NER")
    elif use_bert_multilingual:
        tagger_ner = load_bert_ner("Davlan/bert-base-multilingual-cased-ner-hrl")
    else:
        tagger_ner = SequenceTagger.load("ner-fast")

    start = time()
#   mentions_dataset, n_mentions = mention_detection.find_mentions(docs, tagger_ner)
    mentions_dataset, n_mentions = mention_detection.find_mentions(
       docs,
       (use_bert_base_cased or use_bert_large_cased or use_bert_base_uncased or use_bert_large_uncased or use_bert_multilingual),
       process_sentences,
       split_docs_value,
       tagger_ner)
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

    evaluate_predictions.evaluate(predictions)
