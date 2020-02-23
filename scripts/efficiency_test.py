import requests
from REL.training_datasets import TrainingEvaluationDatasets

base_url = "C:/Users/mickv/Google Drive/projects/entity_tagging/deep-ed/data/"
wiki_version = "wiki_2019/"
datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()['aida_testA']

docs = {}
for i, doc in enumerate(datasets):
    if i == 30:
        break
    sentences = []
    for x in datasets[doc]:
        if x['sentence'] not in sentences:
            sentences.append(x['sentence'])
    text = '. '.join([x for x in sentences])

    # Demo script that can be used to query the API.
    myjson = {
        "text": text,
        "spans": [
            # {"start": 41, "length": 16}
        ],
    }
    print('----------------------------')
    print('Input API:')
    print(myjson)

    print('Output API:')
    print(requests.post("http://localhost:5555", json=myjson).json())
    print('----------------------------')