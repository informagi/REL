import json
import os
from datetime import datetime
from time import time

import numpy as np

import torch

from .consts import NULL_ID_FOR_COREF


def flatten_list_of_lists(lst):
    return [elem for sublst in lst for elem in sublst]


def extract_clusters(gold_clusters):
    gold_clusters = [
        tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m)
        for gc in gold_clusters.tolist()
    ]
    return gold_clusters


def extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[tuple(mention)] = gc
    return mention_to_gold


def extract_clusters_for_decode(mention_to_antecedent, pems_subtoken):
    """
    Args:
        pems (list): E.g., [(2,3), (8,11), ...]
    """

    mention_to_antecedent = sorted(mention_to_antecedent)
    mention_to_cluster = {}
    clusters = []
    for mention, antecedent in mention_to_antecedent:
        if (mention in pems_subtoken) or (antecedent in pems_subtoken):
            if antecedent in mention_to_cluster:
                cluster_idx = mention_to_cluster[antecedent]
                clusters[cluster_idx].append(mention)
                mention_to_cluster[mention] = cluster_idx

            else:
                cluster_idx = len(clusters)
                mention_to_cluster[mention] = cluster_idx
                mention_to_cluster[antecedent] = cluster_idx
                clusters.append([antecedent, mention])
    clusters = [tuple(cluster) for cluster in clusters]
    return clusters, mention_to_cluster


def ce_extract_clusters_for_decode_with_one_mention_per_pem(
    starts, end_offsets, coref_logits, pems_subtoken, flag_use_threshold
):
    """

    Args:
        - flag_use_threshold:
            True: Default. If PEM does not meet a threshold (default: 0), then all mentions are ignored. The threshold is stored in final element of each row of coref_logits.
            False: Ignore threshold, pick the highest logit EEM for each PEM.
    Updates:
      - 220302: Created
    """
    if flag_use_threshold:
        max_antecedents = np.argmax(
            coref_logits, axis=1
        ).tolist()  # HJ: 220225: mention_to_antecedents takes max score. We have at most two predicted EEMs (one is coreference is PEM case, and the other is antecedent is PEM case).
    else:
        max_antecedents = np.argmax(
            coref_logits[:, :-1], axis=1
        ).tolist()  # HJ: 220225: mention_to_antecedents takes max score. We have at most two predicted EEMs (one is coreference is PEM case, and the other is antecedent is PEM case).

    # Create {(ment, antecedent): logits} dict
    mention_antecedent_to_coreflogit_dict = {
        (
            (int(start), int(end)),
            (int(starts[max_antecedent]), int(end_offsets[max_antecedent])),
        ): logit[max_antecedent]
        for start, end, max_antecedent, logit in zip(
            starts, end_offsets, max_antecedents, coref_logits
        )
        if max_antecedent < len(starts)
    }
    # 220403: Drop if key has the same start and end pos for anaphora and antecedent
    mention_antecedent_to_coreflogit_dict = {
        k: v for k, v in mention_antecedent_to_coreflogit_dict.items() if k[0] != k[1]
    }
    if len(mention_antecedent_to_coreflogit_dict) == 0:
        return []

    # Select the ment-ant pair containing the PEM

    mention_antecedent_to_coreflogit_dict_with_pem = {
        (m, a): logit
        for (m, a), logit in mention_antecedent_to_coreflogit_dict.items()
        if (m in pems_subtoken) or (a in pems_subtoken)
    }
    if len(mention_antecedent_to_coreflogit_dict_with_pem) == 0:
        return []

    # Select the max score
    _max_logit = max(mention_antecedent_to_coreflogit_dict_with_pem.values())
    if flag_use_threshold and (_max_logit <= 0):
        print(f"WARNING: _max_logit = {_max_logit}")
    # _max_logit = _max_logit if _max_logit > 0 else 0 # HJ: 220302: If we set a threshold, then this does not work.
    assert (
        coref_logits[-1][-1] == 0
    ), f"The threshold should be 0. If you set your threshold, then the code above should be fixed."
    # Select the pair with max score
    mention_to_antecedent_max_pem = {
        ((m[0], m[1]), (a[0], a[1]))
        for (m, a), logit in mention_antecedent_to_coreflogit_dict_with_pem.items()
        if logit == _max_logit
    }
    assert (
        len(mention_to_antecedent_max_pem) <= 1
    ), f"Two or more mentions have the same max score: {mention_to_antecedent_max_pem}"

    predicted_clusters, _ = extract_clusters_for_decode(
        mention_to_antecedent_max_pem, pems_subtoken
    )  # TODO: 220302: Using `extract_clusters_for_decode` here is redundant.
    return predicted_clusters


def mask_tensor(t, mask):
    t = t + ((1.0 - mask.float()) * -10000.0)
    t = torch.clamp(t, min=-10000.0, max=10000.0)
    return t
