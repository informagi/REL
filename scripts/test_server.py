import os
import requests

# Script for testing the implementation of the conversational entity linking API
#
# To run:
#
#    python .\src\REL\server.py $REL_BASE_URL wiki_2019
# or
#    python .\src\REL\server.py $env:REL_BASE_URL wiki_2019
#
# Set $REL_BASE_URL to where your data are stored (`base_url`)
#
# These paths must exist:
# - `$REL_BASE_URL/bert_conv`
# - `$REL_BASE_URL/s2e_ast_onto `
#
# (see https://github.com/informagi/conversational-entity-linking-2022/tree/main/tool#step-1-download-models)
#


host = 'localhost'
port = '5555'

text1 = {
    "text": "REL is a modular Entity Linking package that can both be integrated in existing pipelines or be used as an API.",
    "spans": []
}

conv1 = {
    "text" : [
        {
            "speaker":
            "USER",
            "utterance":
            "I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.",
        },
        {
            "speaker": "SYSTEM",
            "utterance": "Some people are allergic to histamine in tomatoes.",
        },
        {
            "speaker":
            "USER",
            "utterance":
            "Talking of food, can you recommend me a restaurant in my city for our anniversary?",
        },
    ]
}


for endpoint, myjson in (
        ('', text1), 
        ('conversation/', conv1)
    ):
    print('Input API:')
    print(myjson)
    print()
    print('Output API:')
    print(requests.post(f"http://{host}:{port}/{endpoint}", json=myjson).json())
    print('----------------------------')

