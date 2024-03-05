import os
import re


UNUSED = -1


def get_gold_data(doc, base_url):
    GOLD_DATA_FILE = os.path.join(base_url, "generic/test_datasets/AIDA/AIDA-YAGO2-dataset.tsv")
    entities = []

    in_file = open(GOLD_DATA_FILE, "r")
    for line in in_file:
        if line.startswith(f"-DOCSTART- ({doc} "):
            break
    for line in in_file:
        if line.startswith(f"-DOCSTART- "):
            break
        fields = line.strip().split("\t")
        if len(fields) > 3:
            if fields[1] == "B":
                entities.append([fields[2], fields[3]])
    in_file.close()
    return entities


def md_match(gold_entities, predicted_entities, predicted_links, gold_i, predicted_i):
    return gold_entities[gold_i][0].lower() == predicted_entities[predicted_i][0].lower()


def el_match(gold_entities, predicted_entities, predicted_links, gold_i, predicted_i):
    return(gold_entities[gold_i][0].lower() == predicted_entities[predicted_i][0].lower() and
           gold_entities[gold_i][1].lower() == predicted_entities[predicted_i][1].lower())


def find_correct_els(gold_entities, predicted_entities, gold_links, predicted_links):
    for gold_i in range(0, len(gold_entities)):
        if gold_links[gold_i] == UNUSED:
            for predicted_i in range(0, len(predicted_entities)):
                if (predicted_links[predicted_i] == UNUSED and 
                    el_match(gold_entities, predicted_entities, predicted_links, gold_i, predicted_i)):
                    gold_links[gold_i] = predicted_i
                    predicted_links[predicted_i] = gold_i
    return gold_links, predicted_links


def find_correct_mds(gold_entities, predicted_entities, gold_links, predicted_links):
    for gold_i in range(0, len(gold_entities)):
        if gold_links[gold_i] == UNUSED:
            for predicted_i in range(0, len(predicted_entities)):
                if (predicted_links[predicted_i] == UNUSED and 
                    md_match(gold_entities, predicted_entities, predicted_links, gold_i, predicted_i)):
                    gold_links[gold_i] = predicted_i
                    predicted_links[predicted_i] = gold_i
    return gold_links, predicted_links



def compare_entities(gold_entities, predicted_entities):
    gold_links = len(gold_entities) * [UNUSED]
    predicted_links = len(predicted_entities) * [UNUSED]
    gold_links, predicted_links = find_correct_els(gold_entities, predicted_entities, gold_links, predicted_links)
    gold_links, predicted_links = find_correct_mds(gold_entities, predicted_entities, gold_links, predicted_links)
    return gold_links, predicted_links


def count_entities(gold_entities, predicted_entities, gold_links, predicted_links):
    """ returns: - correct: number of entities correctly identified and correctly linked
                 - wrong_md: number of entities identified but wrong
                 - wrong_el: number of entities correctly identified but incorrectly linked
                 - missed: number of gold standard entities not found and not linked
    """
    correct = 0
    wrong_md = 0
    wrong_el = 0
    missed = 0
    for predicted_i in range(0, len(predicted_links)):
        if predicted_links[predicted_i] == UNUSED:
            wrong_md += 1
        elif predicted_entities[predicted_i][1] == gold_entities[predicted_links[predicted_i]][1]:
            # assumption: predicted_entities[predicted_i][0] == gold_entities[predicted_links[predicted_i]][0]
            correct += 1
        else:
            wrong_el += 1
    for gold_i in range(0, len(gold_links)):
        if gold_links[gold_i] == UNUSED:
            missed += 1
    return correct, wrong_md, wrong_el, missed


def compare_and_count_entities(gold_entities, predicted_entities):
    gold_links, predicted_links = compare_entities(gold_entities, predicted_entities)
    return count_entities(gold_entities, predicted_entities, gold_links, predicted_links)


def compute_md_scores(correct_all, wrong_md_all, wrong_el_all, missed_all):
    correct_md_all = correct_all + wrong_el_all
    if correct_md_all > 0:
        precision_md = correct_md_all / (correct_md_all + wrong_md_all)
        recall_md = correct_md_all / (correct_md_all + missed_all)
        f1_md = 2 * precision_md * recall_md / ( precision_md + recall_md )
    else:
        precision_md = 0
        recall_md = 0
        f1_md = 0
    return precision_md, recall_md, f1_md


def compute_el_scores(correct_all, wrong_md_all, wrong_el_all, missed_all):
    """ reported el_scores are combined md plus el scores """
    if correct_all > 0:
        precision_el = correct_all / (correct_all + wrong_md_all + wrong_el_all)
        recall_el = correct_all / (correct_all + wrong_el_all + missed_all)
        f1_el = 2 * precision_el * recall_el / ( precision_el + recall_el )
    else:
        precision_el = 0.0
        recall_el = 0
        f1_el = 0
    return precision_el, recall_el, f1_el


def print_scores(correct_all, wrong_md_all, wrong_el_all, missed_all):
    precision_md, recall_md, f1_md = compute_md_scores(correct_all, wrong_md_all, wrong_el_all, missed_all)
    precision_el, recall_el, f1_el = compute_el_scores(correct_all, wrong_md_all, wrong_el_all, missed_all)
    print("Results: PMD RMD FMD PEL REL FEL: ", end="")
    print(f"{precision_md:0.1%} {recall_md:0.1%} {f1_md:0.1%} | ",end="")
    print(f"{precision_el:0.1%} {recall_el:0.1%} {f1_el:0.1%}")
    return precision_md, recall_md, f1_md, precision_el, recall_el, f1_el


def evaluate(predictions, base_url):
    correct_all = 0
    wrong_md_all = 0
    wrong_el_all = 0
    missed_all = 0
    for doc in predictions:
        gold_entities = get_gold_data(doc, base_url)
        predicted_entities = []
        for mention in predictions[doc]:
            predicted_entities.append([mention["mention"], mention["prediction"]])
        correct, wrong_md, wrong_el, missed = compare_and_count_entities(gold_entities, predicted_entities)
        correct_all += correct
        wrong_md_all += wrong_md
        wrong_el_all += wrong_el
        missed_all += missed
    print_scores(correct_all, wrong_md_all, wrong_el_all, missed_all)
