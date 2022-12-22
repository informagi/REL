from REL.evaluate_predictions import compare_and_count_entities, print_scores


def test_perfect():
    gold_entities = [ [ "1", "1" ] ]
    predicted_entities = [ [ "1", "1" ] ]
    correct, wrong_md, wrong_el, missed = compare_and_count_entities(gold_entities, predicted_entities)
    precision_md, recall_md, f1_md, precision_el, recall_el, f1_el = print_scores(correct, wrong_md, wrong_el, missed)
    assert [precision_md, recall_md, f1_md, precision_el, recall_el, f1_el] == [100, 100, 100, 100, 100, 100], "should be perfect MD and perfect EL"


def test_el_wrong():
    gold_entities = [ [ "1", "1" ] ]
    predicted_entities = [ [ "1", "0" ] ]
    correct, wrong_md, wrong_el, missed = compare_and_count_entities(gold_entities, predicted_entities)
    precision_md, recall_md, f1_md, precision_el, recall_el, f1_el = print_scores(correct, wrong_md, wrong_el, missed)
    assert [precision_md, recall_md, f1_md, precision_el, recall_el, f1_el] == [100, 100, 100, 0, 0, 0], "should be perfect MD and failed EL"


def test_md_wrong():
    gold_entities = [ [ "1", "1" ] ]
    predicted_entities = [ [ "0", "1" ] ]
    correct, wrong_md, wrong_el, missed = compare_and_count_entities(gold_entities, predicted_entities)
    precision_md, recall_md, f1_md, precision_el, recall_el, f1_el = print_scores(correct, wrong_md, wrong_el, missed)
    assert [precision_md, recall_md, f1_md, precision_el, recall_el, f1_el] == [0, 0, 0, 0, 0, 0], "should be failed MD and failed EL"


def test_combined():
    gold_entities = [ [ "1", "1" ], [ "1", "1" ], [ "2", "2" ] ]
    predicted_entities = [ [ "0", "0" ], [ "0", "1" ], [ "1", "0" ], [ "1", "1" ] ]
    correct, wrong_md, wrong_el, missed = compare_and_count_entities(gold_entities, predicted_entities)
    precision_md, recall_md, f1_md, precision_el, recall_el, f1_el = print_scores(correct, wrong_md, wrong_el, missed)
    assert [precision_md, recall_md, f1_md, precision_el, recall_el, f1_el] == [100/2, 100*2/3, 100*4/7, 100/4, 100/3, 100*2/7], "should be various scores"

