TOKEN = 'fb7d26fa7976b605905934dbd48ac892'
FIELDS = "entities,sentiment,facts"
HOST = "nl.diffbot.com"
PORT = "80"

import json
import requests

def get_request(payload):
    res = requests.post("http://{}/v1/?fields={}&token={}".format(HOST, FIELDS, TOKEN), json=payload)
    return res.json()

content = "Protectively Iran will protest to the International Court of Justice at the Hague"

res = get_request({
    "content": content,
    "lang": "en",
    "format": "plain text"
})

# {'sentiment': -0.56536525, 'entities': [{'name': 'Iran', 'diffbotUri': 'https://diffbot.com/entity/AjQZxbxbvMoWkkLfL9-INZw', 'confidence': 0.973663, 'salience': 0.9886276, 
# 'sentiment': -0.8488826, 'isCustom': False, 'allUris': ['http://www.wikidata.org/entity/Q794'], 'allTypes': [{'name': 'location', 'diffbotUri': 
# 'https://diffbot.com/entity/XiCyWUm4lNziqgLHx47iAIQ', 'dbpediaUri': 'http://dbpedia.org/ontology/Place'}, {'name': 'administrative area', 'diffbotUri': 'https://diffbot.com/entity/XcTIu1tWKPouIa6qZtSpc4A', 
# 'dbpediaUri': 'http://dbpedia.org/ontology/PopulatedPlace'}, {'name': 'country', 'diffbotUri': 'https://diffbot.com/entity/X9bmxGo8aNgCDUxsGSFl_9A', 'dbpediaUri': 
# 'http://dbpedia.org/ontology/Country'}], 'mentions': [{'text': 'Iran', 'beginOffset': 13, 'endOffset': 17, 'confidence': 0.973663}], 'location': {'latitude': 32.0, 'longitude': 53.0, 'precision': 1283.8204}}, 
# {'name': 'The Hague', 'diffbotUri': 'https://diffbot.com/entity/Ao6hwtNt-PUSIpfh1PRfASg', 'confidence': 0.9053129, 'salience': 0.62889826, 'sentiment': 0.0, 'isCustom': False, 'allUris': 
# ['http://www.wikidata.org/entity/Q36600'], 'allTypes': [{'name': 'location', 'diffbotUri': 'https://diffbot.com/entity/XiCyWUm4lNziqgLHx47iAIQ', 'dbpediaUri': 
# 'http://dbpedia.org/ontology/Place'}, {'name': 'administrative area', 'diffbotUri': 'https://diffbot.com/entity/XcTIu1tWKPouIa6qZtSpc4A', 'dbpediaUri': 
# 'http://dbpedia.org/ontology/PopulatedPlace'}, {'name': 'city', 'diffbotUri': 'https://diffbot.com/entity/XzdJrGHiyMWu0XbSn101rFA', 'dbpediaUri': 'http://dbpedia.org/ontology/City'}], 
# 'mentions': [{'text': 'Hague', 'beginOffset': 76, 'endOffset': 81, 'confidence': 0.9053129}], 'location': {'latitude': 52.084167, 'longitude': 4.3175, 'precision': 9.899495}}], 'facts': []}
