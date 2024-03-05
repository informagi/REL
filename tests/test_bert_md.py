#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import os
import pytest

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from REL.mention_detection import MentionDetection
from REL.ner.bert_wrapper import load_bert_ner


@pytest.mark.skipif( 
    os.getenv("GITHUB_ACTIONS")=='true', reason="No way of testing this on Github actions."
)
def test_md():
    tagger_ner = load_bert_ner("dslim/bert-base-NER")
    process_sentences = False
    split_docs_value = 0
    base_url = os.path.dirname(__file__) + "/../src/data/"
    wiki_version = "wiki_2019"
    md = MentionDetection(base_url, wiki_version)

    # first test case: repeating sentences
    sample1 = {"test_doc": ["Fox, Fox. Fox.", []]}
    resulting_spans1 = {(0, 3), (5, 3), (10, 3)}
    predictions = md.find_mentions(sample1, process_sentences, split_docs_value, tagger_ner)
    predicted_spans = []
    for i in range(0, 1):
        p = {
            (m["pos"], m["end_pos"] - m["pos"]) for m in predictions[i]["test_doc"]
        }
        predicted_spans.extend(list(p))
    predicted_spans = set(predicted_spans)
    assert resulting_spans1 == predicted_spans

    # second test case: excessive whitespace
    sample2 = {"test_doc": ["Fox,                Fox,                   Fox.", []]}
    resulting_spans2 = {(0, 3), (20, 3), (43, 3)}
    predictions = md.find_mentions(sample2, process_sentences, split_docs_value, tagger_ner)
    predicted_spans = {
        (m["pos"], m["end_pos"] - m["pos"]) for m in predictions[0]["test_doc"]
    }
    assert resulting_spans2 == predicted_spans
