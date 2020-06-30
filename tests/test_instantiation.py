#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import torch

from REL.entity_disambiguation import EntityDisambiguation
from REL.mention_detection import MentionDetection
from REL.mulrel_ranker import MulRelRanker, PreRank
from REL.ner import Cmns


def test_entity_disambiguation_instantiation():
    return EntityDisambiguation(
        Path(__file__).parent,
        "wiki_test",
        {
            "mode": "eval",
            "model_path": Path(__file__).parent / "wiki_test" / "generated" / "model",
        },
    )


def test_cmns_instantiation():
    return Cmns(Path(__file__).parent, "wiki_test")


def test_mention_detection_instantiation():
    return MentionDetection(Path(__file__).parent, "wiki_test")


def test_prerank_instantiation():
    # NOTE: this is basically just a blank constructor; if this fails, something is
    # seriously wrong
    return PreRank({})


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
    return MulRelRanker(config, torch.device("cpu"))
