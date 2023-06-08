# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-13 16:15:17
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-13 16:18:10
"""Evaluate the detected communities using the element-centric similarity."""
# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats
from embcom.birch import CosineBirch

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    metric = params["metric"]
    model_name = params["model_name"]
    n_clusters = snakemake.params["n_clusters"]
else:
    emb_file = "../../data/lfr/embedding/n~100000_k~100_tau~3_tau2~1_minc~50_mu~0.70_sample~2_model_name~linearized-node2vec_window_length~10_dim~64.npz"
    com_file = "../../data/lfr/networks/node_n~100000_k~100_tau~3_tau2~1_minc~50_mu~0.70_sample~2.npz"
    model_name = "linearized-node2vec"
    output_file = "unko"
    n_clusters = "data"
    metric = "cosine"


# Load emebdding
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
memberships = pd.read_csv(com_file)["membership"].values.astype(int)
# Remove nan embedding
remove = np.isnan(np.array(np.sum(emb, axis=1)).reshape(-1))
keep = np.where(~remove)[0]
n_nodes = emb.shape[0]
emb = emb[keep, :]
memberships = memberships[keep]

emb_copy = emb.copy()

results = {}
for dimThreshold in [True, False]:
    for normalize in [True, False]:
        emb = emb_copy.copy()
        if model_name == "nonbacktracking":
            norm = np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
            idx = np.argmax(norm)
            threshold = np.sqrt(norm[idx])
            keep_dims = norm >= threshold
            keep_dims[idx] = False
            if any(keep_dims) is False:
                keep_dims[idx] = True
            emb = emb[:, keep_dims]

        if normalize:
            norm = np.array(np.linalg.norm(emb, axis=0)).reshape(-1)
            emb = np.einsum("ij,j->ij", emb, 1 / np.maximum(norm, 1e-32))

        # Evaluate
        group_ids = CosineBirch(
            emb=emb, group_ids=memberships, n_clusters=n_clusters, metric=metric
        )
        # group_ids = KMeans(emb, memberships, metric=metric)

        key = f"normalize~{normalize}_dimThreshold~{dimThreshold}"

        # Set the group index to nan if the node has no embedding
        group_ids_ = np.zeros(n_nodes) * np.nan
        group_ids_[keep] = group_ids
        results[key] = group_ids_

# %%
# Save
#
np.savez(output_file, **results)
