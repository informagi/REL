import requests

IP_ADDRESS = "http://127.0.0.1"
PORT = "1235"
text_doc = "If you're going to try, go all the way - Charles Bukowski"

# Example EL.
document = {
    "text": text_doc,
    "spans": []
}

# Example ED.
# document = {
#     "text": text_doc,
#     "spans": [(41, 16)]
# }


API_result = requests.post("{}:{}".format(IP_ADDRESS, PORT), json=document).json()