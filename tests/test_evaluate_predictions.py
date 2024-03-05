from scripts.evaluate_predictions import compare_and_count_entities, print_scores
import pytest


def test_perfect():
    gold_entities = [ [ "1", "1" ] ]
    predicted_entities = [ [ "1", "1" ] ]
    counts = compare_and_count_entities(gold_entities, predicted_entities)
    scores = print_scores(*counts)
    assert list(scores) == [1, 1, 1, 1, 1, 1], "should be perfect MD and perfect EL"


def test_el_wrong():
    gold_entities = [ [ "1", "1" ] ]
    predicted_entities = [ [ "1", "0" ] ]
    counts = compare_and_count_entities(gold_entities, predicted_entities)
    scores = print_scores(*counts)
    assert list(scores) == [1, 1, 1, 0, 0, 0], "should be perfect MD and failed EL"


def test_md_wrong():
    gold_entities = [ [ "1", "1" ] ]
    predicted_entities = [ [ "0", "1" ] ]
    counts = compare_and_count_entities(gold_entities, predicted_entities)
    scores = print_scores(*counts)
    assert list(scores) == [0, 0, 0, 0, 0, 0], "should be failed MD and failed EL"


def test_combined():
    gold_entities = [ [ "1", "1" ], [ "1", "1" ], [ "2", "2" ] ]
    predicted_entities = [ [ "0", "0" ], [ "0", "1" ], [ "1", "0" ], [ "1", "1" ] ]
    counts = compare_and_count_entities(gold_entities, predicted_entities)
    scores = print_scores(*counts)
    target_scores = [1/2, 2/3, 4/7, 1/4, 1/3, 2/7]
    for index in range(0, len(scores)):
        assert pytest.approx(scores[index], 0.0001) == target_scores[index], "should be various scores"

