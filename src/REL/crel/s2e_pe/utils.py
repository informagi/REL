import json
import os
from datetime import datetime
from time import time

import numpy as np

# import git
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


# def extract_clusters_for_decode(mention_to_antecedent):
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


# def write_meta_data(output_dir, args):
#     output_path = os.path.join(output_dir, "meta.json")
#     repo = git.Repo(search_parent_directories=True)
#     hexsha = repo.head.commit.hexsha
#     ts = time()
#     print(f"Writing {output_path}")
#     with open(output_path, mode='w') as f:
#         json.dump(
#             {
#                 'git_hexsha': hexsha,
#                 'args': {k: str(v) for k, v in args.__dict__.items()},
#                 'date': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#             },
#             f,
#             indent=4,
#             sort_keys=True)

# def ce_get_start_end_subtoken_num(start_token, end_token, subtoken_map):
#     """
#     Example:
#         ### Input ###
#         start_token, end_token = (2,4)
#         subtoken_map = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # subtoken_map

#         ### Output ###
#         (6, 8)

#     Notes:
#         - This function is used in modeling.py and this util.py
#     """
#     N = len(subtoken_map)
#     start_subtoken = subtoken_map.index(start_token)
#     end_subtoken = (N-1) - subtoken_map[::-1].index(end_token)

#     return start_subtoken, end_subtoken

# def ce_get_pem_ments(predict_file):
#     """
#     Args:
#         args.predict_file: E.g., '../data/data_dir/test.english.jsonlines'
#     Return:
#         - pems  (dict): E.g., {'dialind_0': [[2, 4]], 'dialind_1': [[66, 67]], ...}
#         - ments (dict): E.g., {'dialind_0': [[0, 0], [3, 4], [35, 35], ...}
#     """
#     pems = {}
#     ments = {}
#     with open(predict_file) as f:
#         jsonl = f.readlines()
#         print(len(jsonl))
#         for l in jsonl:
#             l = json.loads(l)
#             assert l['doc_key'] not in pems, 'doc_key should be unique'
#             assert l['doc_key'] not in ments, 'doc_key should be unique'
#             pems[l['doc_key']] = l['pems']
#             ments[l['doc_key']] = l['mentions']
#             # pems.append(l['pems']) # PEM
#             # ments.append(l['mentions']) # mentions

#     # Error check: pem should not be in ments
#     for doc_key in pems: # doc_key: E.g., 'dialind_0'
#         for pem in pems[doc_key]: # pem: E.g., [2, 4]
#             assert pem not in ments[doc_key], f'PEM {pem} should not be in ments. Fix this at 010_preprocess_eval_ConEL.ipynb'

#     return pems, ments
