#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from REL.ner import Cmns, Span


def compare_spans(a: Span, b: Span, fields=(0, 1, 2)):
    for f in fields:
        if a[f] != b[f]:
            return False
    else:
        return True


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

    return compare_spans(predictions, labels)
