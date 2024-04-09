#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import os
import pytest

from REL.ner import Cmns, Span


def compare_spans(a: Span, b: Span, fields=(0, 1, 2)):
    if len(a) != len(b):
        return False
    for f in fields:
        for index in range(0, len(a)):
            if a[index][f] != b[index][f]:
                return (False, a[index][f], b[index][f])
    else:
        return True


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS")=='true', reason="No way of testing this on Github actions."
)
def test_cmns():
    model = Cmns(Path(__file__).parent, "wiki_test", n=5)
    predictions = model.predict("the brown fox jumped over the lazy dog", None)
    labels = [
        Span("the", 0, 3, None, None),
        Span("brown", 4, 9, None, None),
        Span("fox", 10, 13, None, None),
        Span("jumped", 14, 20, None, None),
        Span("over", 21, 25, None, None),
        Span("the", 26, 29, None, None),
        Span("lazy", 30, 34, None, None),
        Span("dog", 35, 38, None, None),
    ]

    assert compare_spans(predictions, labels) == True
