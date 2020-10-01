import requests

IP_ADDRESS = "http://127.0.0.1"
PORT = "5555"

text_docs = [
"""
REPEATING Protectively Iran will protest to the International Court of Justice at the Hague and other global bodies about the U.S.-funded Radio Free Europe , the Iran Daily reported Monday. It quoted Foreign Minister Kamal Kharrazi 
"""
# [Turn 1] I would like a moderately priced restaurant in the west part of town
# [Turn 2] There are 3 moderately priced restaurants in the west part of town; Meghna, Prezzo, and Saint Johns Chop House. How else may I assist you?
# [Turn 3] What is the phone number and address of one of them?
# """,
# """
# [Turn 1] I would like a moderately priced restaurant in the west part of town
# [Turn 2] There are 3 moderately priced restaurants in the west part of town; Meghna, Prezzo, and Saint Johns Chop House. How else may I assist you?
# [Turn 3] What is the phone number and address of one of them?
# [Turn 4] Meghna is located at 205 Victoria Road Chesterton and their number is 01223 727410. Is there anything else I can do for you today?
# """
]

for text_doc in text_docs:
    # Example EL.
    document = {
        "text": text_doc,
        # "spans": []
    }


    API_result = requests.post("{}:{}".format(IP_ADDRESS, PORT), json=document).json()
    for a in API_result:
        print(a)
        print('---')
    print(API_result)
    print('=====')
