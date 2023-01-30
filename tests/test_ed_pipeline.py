#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import os
import pytest

from REL.entity_disambiguation import EntityDisambiguation
from REL.mention_detection import MentionDetection
from REL.ner import Cmns
from REL.utils import process_results


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS")=='true', reason="No way of testing this on Github actions."
)
def test_pipeline():
    base_url = Path(__file__).parent
    wiki_subfolder = "wiki_test"
    sample = {"test_doc": ["the brown fox jumped over the lazy dog", [[10, 3]]]}
    config = {
        "mode": "eval",
        "model_path": f"{base_url}/{wiki_subfolder}/generated/model",
    }

    md = MentionDetection(base_url, wiki_subfolder)
    tagger = Cmns(base_url, wiki_subfolder, n=5)
    model = EntityDisambiguation(base_url, wiki_subfolder, config)

    mentions_dataset, total_mentions = md.format_spans(sample, tagger_ner_name="flair", process_sentences=True)

    predictions, _ = model.predict(mentions_dataset)
    results = process_results(
        mentions_dataset, predictions, sample, include_offset=False
    )

    gold_truth = {"test_doc": [(10, 3, "fox", "Fox", 0.0, 0.0, "NULL")]}

    assert results == gold_truth
