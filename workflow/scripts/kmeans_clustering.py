"""Cluster node embeddings using K-means and save predicted community labels."""
import sys

import numpy as np
import pandas as pd
from sklearn import cluster

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    metric = params["metric"]
    model_name = params["model_name"]
else:
    emb_file = "../../data/multi_partition_model/embedding/n~1000_K~2_cave~50_mu~0.70_sample~1_model_name~leigenmap_window_length~10_dim~64.npz"
    com_file = "../../data/multi_partition_model/networks/node_n~2500_K~2_cave~50_mu~0.70_sample~1.npz"
    model_name = "leigenmap"
    output_file = "unko"
    metric = "cosine"


def run_kmeans(emb, memberships, metric="euclidean"):
    """Run K-means with K inferred from ground-truth community count."""
    n_communities = np.max(memberships) + 1
    # Note: cosine normalization is intentionally disabled (kept for interface compatibility).
    if metric == "cosine":
        X = np.einsum(
            "ij,i->ij", emb, 1 / np.maximum(np.linalg.norm(emb, axis=1), 1e-24)
        )
        X = emb
    kmeans = cluster.KMeans(n_clusters=n_communities, random_state=0).fit(X)
    return kmeans.labels_


# Load embedding and ground-truth memberships
emb_raw = np.load(emb_file)["emb"]
emb_raw = emb_raw.copy(order="C").astype(np.float32)
memberships = pd.read_csv(com_file)["membership"].values.astype(int)

# Remove nodes whose embedding contains NaN
n_nodes = emb_raw.shape[0]
valid_mask = ~np.isnan(emb_raw.sum(axis=1))
valid_indices = np.where(valid_mask)[0]
emb_valid = emb_raw[valid_indices, :]
memberships_valid = memberships[valid_indices]

# Try all combinations of dimension thresholding and column normalization
results = {}
for dim_threshold in [True, False]:
    for normalize in [True, False]:
        emb = emb_valid.copy()

        # For non-backtracking embeddings, discard dimensions below a spectral threshold
        if (model_name == "nonbacktracking") & dim_threshold:
            col_norms = np.linalg.norm(emb, axis=0)
            dominant_dim = np.argmax(col_norms)
            threshold = np.sqrt(col_norms[dominant_dim])
            keep_dims = col_norms >= threshold
            keep_dims[dominant_dim] = False
            if any(keep_dims) is False:
                keep_dims[dominant_dim] = True
            emb = emb[:, keep_dims]

        # Normalize each column to unit norm
        if normalize:
            col_norms = np.linalg.norm(emb, axis=0)
            emb = np.einsum("ij,j->ij", emb, 1 / np.maximum(col_norms, 1e-32))

        predicted_labels = run_kmeans(emb, memberships_valid, metric=metric)

        # Place predictions back into a full-size array (NaN for removed nodes)
        full_labels = np.full(n_nodes, np.nan)
        full_labels[valid_indices] = predicted_labels

        key = f"normalize~{normalize}_dimThreshold~{dim_threshold}"
        results[key] = full_labels

np.savez(output_file, **results)
