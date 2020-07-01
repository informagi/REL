"""
Hacky - cli call to rel with string parameter
python3 text2tag
"""

import json
import sys

import requests

IP_ADDRESS = "http://gem.cs.ru.nl/api"
PORT = "80"


def main(filename):

    fi = open(filename, "r")
    document = {
        "text": "",
        "spans": [],
    }
    for q in fi:
        q = q.strip()
        document["text"] = q
        r = requests.post("{}:{}".format(IP_ADDRESS, PORT), json=document)
        if r.text == "":
            API_result = []
        else:
            API_result = r.json()

        print("{}: {}".format(q, API_result))

    fi.close()


####### Entry:

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\n python {} textfile".format(__file__))
    else:
        main(sys.argv[1])
