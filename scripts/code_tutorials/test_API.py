import requests

IP_ADDRESS = "http://127.0.0.1"
PORT = "5555"
# text_doc = "If you're going to try, go all the way - Charles Bukowski"
# Example ED.
# document = {"text": text_doc, "spans": [(41, 16)]}

# text_doc = "REPEATING Protectively Iran will protest to the International Court of Justice at the Hague and other global bodies about the U.S.-funded Radio Free Europe , the Iran Daily reported Monday. It quoted Foreign Minister Kamal Kharrazi "
text_doc = "This simple text is for testing the communication"
# Example EL.
document = {
    "text": text_doc
}


API_result = requests.post("{}:{}".format(IP_ADDRESS, PORT), json=document).json()
# print(API_result)