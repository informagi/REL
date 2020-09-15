import argparse
import logging

import blink.main_dense as main_dense

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models_path = "/users/vanhulsm/Desktop/projects/BLINK/models/"  # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "biencoder_model": models_path + "biencoder_wiki_large.bin",
    "biencoder_config": models_path + "biencoder_wiki_large.json",
    "entity_catalogue": models_path + "entity.jsonl",
    "entity_encoding": models_path + "all_entities_large.t7",
    "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
    "crossencoder_config": models_path + "crossencoder_wiki_large.json",
    "fast": True,  # set this to be true if speed is a concern
    "output_path": "logs/",  # logging directory
    "top_k": 1,
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args)

data_to_link = [
    {
        "id": 0,
        "label": "unknown",
        "label_id": -1,
        "context_left": "".lower(),
        "mention": "Shakespeare".lower(),
        "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
    },
    {
        "id": 1,
        "label": "unknown",
        "label_id": -1,
        "context_left": "Shakespeare's account of the Roman general".lower(),
        "mention": "Julius Caesar".lower(),
        "context_right": "'s murder by his friend Brutus is a meditation on duty.".lower(),
    },
]

# TODO: Add ED
(
    _,
    _,
    _,
    _,
    _,
    predictions,
    scores,
) = main_dense.run(args, logger, *models, test_data=data_to_link)


print("============")
print(predictions)
print("---")
print(scores)
