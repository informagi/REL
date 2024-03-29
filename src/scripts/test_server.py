import os

import requests

# Script for testing the implementation of the conversational entity linking API
#
# To run:
#
#    python .\src\REL\server.py $REL_BASE_URL wiki_2019 --ner-model ner-fast ner-fast-with-lowercase
# or
#    python .\src\REL\server.py $env:REL_BASE_URL wiki_2019 --ner-model ner-fast ner-fast-with-lowercase
#
# Set $REL_BASE_URL to where your data are stored (`base_url`)
#
# These paths must exist:
# - `$REL_BASE_URL/bert_conv`
# - `$REL_BASE_URL/s2e_ast_onto `
#
# (see https://github.com/informagi/conversational-entity-linking-2022/tree/main/tool#step-1-download-models)
#


host = "localhost"
port = "5555"

items = (
    {
        "endpoint": "",
        "payload": {
            "tagger": "ner-fast",
            "text": "REL is a modular Entity Linking package that can both be integrated in existing pipelines or be used as an API.",
            "spans": [],
        },
    },
    {
        "endpoint": "ne",
        "payload": {
            "tagger": "ner-fast-with-lowercase",
            "text": "REL is a modular Entity Linking package that can both be integrated in existing pipelines or be used as an API.",
            "spans": [],
        },
    },
    {
        "endpoint": "ne",
        "payload": {
            "tagger": "ner-fast",
            "text": "If you're going to try, go all the way - Charles Bukowski.",
            "spans": [(41, 16)],
        },
    },
    {
        "endpoint": "conv",
        "payload": {
            "tagger": "default",
            "text": [
                {
                    "speaker": "USER",
                    "utterance": "I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.",
                },
                {
                    "speaker": "SYSTEM",
                    "utterance": "Some people are allergic to histamine in tomatoes.",
                },
                {
                    "speaker": "USER",
                    "utterance": "Talking of food, can you recommend me a restaurant in my city for our anniversary?",
                },
            ],
        },
    },
    {
        "endpoint": "ne_concept",
        "payload": {},
    },
    {
        "endpoint": "this-endpoint-does-not-exist",
        "payload": {},
    },
    {
        "endpoint": "",
        "payload": {
            "text": "Hello world.",
            "this-argument-does-not-exist": None,
        },
    },
)

for item in items:
    endpoint = item["endpoint"]
    payload = item["payload"]

    print("Request body:")
    print(payload)
    print()
    print("Response:")
    print(requests.post(f"http://{host}:{port}/{endpoint}", json=payload).json())
    print()
    print("----------------------------")
    print()
