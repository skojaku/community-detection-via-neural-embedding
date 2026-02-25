# -*- coding: utf-8 -*-
"""Evaluate detected communities using NMI and element-centric similarity."""

import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.cluster import normalized_mutual_info_score

if "snakemake" in sys.modules:
    detected_group_file = snakemake.input["detected_group_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
else:
    com_file = "../../data/multi_partition_model/networks/node_n~10000_K~2_cave~50_mu~0.50_sample~0.npz"
    detected_group_file = "../../data/multi_partition_model/communities/clus_n~10000_K~2_cave~50_mu~0.50_sample~0_model_name~leigenmap_window_length~10_dim~0_metric~cosine_clustering~voronoi.npz"
    output_file = "output.csv"


def calc_esim(y_true, y_pred):
    """Compute element-centric similarity between two partitions.

    Uses a chance-corrected formula: (S - S_rand) / (1 - S_rand).
    """
    _, y_true = np.unique(y_true, return_inverse=True)
    _, y_pred = np.unique(y_pred, return_inverse=True)

    num_true = len(np.unique(y_true))
    num_pred = len(np.unique(y_pred))
    K = max(num_true, num_pred)
    N = len(y_true)

    # Membership matrices: rows = nodes, columns = communities
    U_true = sparse.csr_matrix(
        (np.ones(N), (np.arange(N), y_true)), shape=(N, K)
    )
    U_pred = sparse.csr_matrix(
        (np.ones(N), (np.arange(N), y_pred)), shape=(N, K)
    )

    # Community sizes
    size_true = np.asarray(U_true.sum(axis=0)).ravel()
    size_pred = np.asarray(U_pred.sum(axis=0)).ravel()

    # Confusion matrix and its random expectation
    confusion = (U_true.T @ U_pred).toarray()
    confusion_rand = np.outer(size_true, size_pred) / N

    # Normalization factor: 1 / max(size_true_i, size_pred_j) for each pair
    Q = np.maximum(
        size_true[:, None] * np.ones((1, K)),
        np.ones((K, 1)) * size_pred[None, :],
    )
    Q = 1.0 / np.maximum(Q, 1)

    S = np.sum(Q * (confusion ** 2)) / N
    S_rand = np.sum(Q * (confusion_rand ** 2)) / N

    return (S - S_rand) / (1 - S_rand)


def parse_params(key_string, sep="~"):
    """Parse 'param~value' pairs from an underscore-delimited key string.

    For example, 'n~100_K~3' yields {'n': '100', 'K': '3'}.
    """
    params = {}
    for token in key_string.split("_"):
        if sep not in token:
            continue
        name, value = token.split(sep, 1)
        params[name] = value
    return params


# Load ground-truth memberships and detected communities
memberships = pd.read_csv(com_file)["membership"].values.astype(int)
detected_groups = np.load(detected_group_file)

# Score each detected partition against ground truth
SCORE_FUNCTIONS = {
    "nmi": normalized_mutual_info_score,
    "esim": calc_esim,
}

results = []
for key, group_ids in detected_groups.items():
    valid = ~np.isnan(group_ids)
    memberships_valid = memberships[valid]
    group_ids_valid = group_ids[valid]

    params = parse_params(key)
    for score_type, score_fn in SCORE_FUNCTIONS.items():
        score = score_fn(memberships_valid, group_ids_valid)
        results.append({"score": score, "score_type": score_type, **params})

pd.DataFrame(results).to_csv(output_file, index=False)
