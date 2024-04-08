import re

from flair.models import SequenceTagger
from REL.ner.bert_wrapper import load_bert_ner

taggers_ner = {
    "flair": "flair/ner-english-fast",
    "bert_base_cased": "dslim/bert-base-NER",
    "bert_base_uncased": "dslim/bert-base-NER-uncased",
    "bert_large_cased": "dslim/bert-large-NER",
    "bert_large_uncased": "Jorgeutd/bert-large-uncased-finetuned-ner",
    "bert_multilingual": "Davlan/bert-base-multilingual-cased-ner-hrl"
}


def set_tagger_ner(tagger_ner_name):
    if re.search("^flair", tagger_ner_name):
        tagger_ner = SequenceTagger.load(taggers_ner[tagger_ner_name])
    elif re.search("^bert", tagger_ner_name):
        tagger_ner = load_bert_ner(taggers_ner[tagger_ner_name])
    else:
        raise Exception(f"unknown tagger name: {tagger_ner_name}")

    return tagger_ner
