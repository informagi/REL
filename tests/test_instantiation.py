#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import os
import pytest
import torch

from REL.entity_disambiguation import EntityDisambiguation
from REL.mention_detection import MentionDetection
from REL.mulrel_ranker import MulRelRanker, PreRank
from REL.ner import Cmns


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS")=='true', reason="No way of testing this on Github actions."
)
def test_entity_disambiguation_instantiation():
    assert True == bool(EntityDisambiguation(
        Path(__file__).parent,
        "wiki_test",
        {
            "mode": "eval",
            "model_path": Path(__file__).parent / "wiki_test" / "generated" / "model",
        },
    ))


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS")=='true', reason="No way of testing this on Github actions."
)
def test_cmns_instantiation():
    assert True == bool(Cmns(Path(__file__).parent, "wiki_test"))


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS")=='true', reason="No way of testing this on Github actions."
)
def test_mention_detection_instantiation():
    assert True == bool(MentionDetection(Path(__file__).parent, "wiki_test"))


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS")=='true', reason="No way of testing this on Github actions."
)
def test_prerank_instantiation():
    # NOTE: this is basically just a blank constructor; if this fails, something is
    # seriously wrong
    assert True == bool(PreRank({}))


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS")=='true', reason="No way of testing this on Github actions."
)
def test_mulrel_ranker_instantiation():
    # minimal config to make the constructor run
    config = {
        "emb_dims": 300,
        "hid_dims": 100,
        "dropout_rate": 0.3,
        "n_rels": 3,
        "use_local": True,
        "use_pad_ent": True,
    }
    assert True == bool(MulRelRanker(config, torch.device("cpu")))
