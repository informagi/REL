#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import os

from flair.models import SequenceTagger

from REL.mention_detection import MentionDetection


def test_md():
    # return standard Flair tagger + mention detection object
    tagger_ner_name = "flair"
    tagger_ner = SequenceTagger.load("ner-fast")
    process_sentences = True
    split_docs_value = 0
    base_url = os.path.dirname(__file__) + "/../data/"
    wiki_version = "wiki_2019"
    md = MentionDetection(base_url, wiki_version)

    # first test case: repeating sentences
    sample1 = {"test_doc": [ "Fox. Fox. Fox." , []] }
    resulting_spans1 = {(0, 3), (5, 3), (10, 3)}
    predictions = md.find_mentions(sample1, tagger_ner_name, process_sentences, split_docs_value, tagger_ner)

    predicted_spans = {
        (m["pos"], m["end_pos"] - m["pos"]) for m in predictions[0]["test_doc"]
    }
    assert resulting_spans1 == predicted_spans

    # second test case: excessive whitespace
    sample2 = {"test_doc": ["Fox.                Fox.                   Fox.", []]}
    resulting_spans2 = {(0, 3), (20, 3), (43, 3)}
    predictions = md.find_mentions(sample2, tagger_ner_name, process_sentences, split_docs_value, tagger_ner)
    predicted_spans = {
        (m["pos"], m["end_pos"] - m["pos"]) for m in predictions[0]["test_doc"]
    }
    assert resulting_spans2 == predicted_spans
