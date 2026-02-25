"""Assign nodes to communities via Voronoi (nearest-centroid) clustering in embedding space."""
import sys

import numpy as np
import pandas as pd
from scipy import sparse

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    metric = params["metric"]
    model_name = params["model_name"]
else:
    emb_file = "../../data/lfr/embedding/n~1000_k~100_tau~3_tau2~1_minc~50_mu~0.70_sample~9_model_name~linearized-node2vec_window_length~10_dim~64.npz"
    com_file = "../../data/lfr/networks/node_n~1000_k~100_tau~3_tau2~1_minc~50_mu~0.70_sample~9.npz"
    model_name = "linearized-node2vec"
    output_file = "unko"
    metric = "cosine"


def row_normalize(mat):
    """Normalize rows of a sparse CSR matrix so each row sums to 1.

    Rows that sum to zero remain all zeros.
    """
    row_sums = np.array(mat.sum(axis=1)).ravel().astype(float)
    inv_sums = 1.0 / np.maximum(row_sums, 1e-32)
    return sparse.diags(inv_sums, format="csr") @ mat


def voronoi_assign(emb, memberships, metric="euclidean"):
    """Assign each node to the nearest ground-truth community centroid.

    Centroids are computed as the mean embedding of each community.
    """
    n_nodes = emb.shape[0]
    n_communities = np.max(memberships) + 1

    # Build membership matrix and compute community centroids
    membership_matrix = sparse.csr_matrix(
        (np.ones_like(memberships), (np.arange(n_nodes), memberships)),
        shape=(n_nodes, n_communities),
    )
    membership_matrix = row_normalize(membership_matrix)
    centroids = membership_matrix.T @ emb

    if metric == "cosine":
        emb_normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        centroids_normed = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        return np.argmax(emb_normed @ centroids_normed.T, axis=1)
    elif metric == "dotsim":
        return np.argmax(emb @ centroids.T, axis=1)
    elif metric == "euclidean":
        sq_norm_emb = np.linalg.norm(emb, axis=1) ** 2
        sq_norm_centroids = np.linalg.norm(centroids, axis=1) ** 2
        dist = np.add.outer(sq_norm_emb, sq_norm_centroids) - 2 * emb @ centroids.T
        return np.argmin(dist, axis=1)


# Load embedding and ground-truth memberships
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
memberships = pd.read_csv(com_file)["membership"].values.astype(int)

# Filter out nodes whose embeddings contain NaN
valid_mask = ~np.isnan(emb.sum(axis=1))
valid_indices = np.where(valid_mask)[0]
n_nodes = emb.shape[0]
emb_valid = emb[valid_indices, :]
memberships_valid = memberships[valid_indices]

emb_base = emb_valid.copy()

results = {}
for dim_threshold in [True, False]:
    for normalize in [True, False]:
        emb_iter = emb_base.copy()

        # For nonbacktracking embeddings, discard dimensions below a spectral threshold
        if model_name == "nonbacktracking":
            col_norms = np.linalg.norm(emb_iter, axis=0)
            leading_idx = np.argmax(col_norms)
            threshold = np.sqrt(col_norms[leading_idx])
            keep_dims = col_norms >= threshold
            keep_dims[leading_idx] = False
            if any(keep_dims) is False:
                keep_dims[leading_idx] = True
            emb_iter = emb_iter[:, keep_dims]

        # Optionally normalize columns to unit norm
        if normalize:
            col_norms = np.linalg.norm(emb_iter, axis=0)
            emb_iter = emb_iter / np.maximum(col_norms, 1e-32)

        cluster_ids = voronoi_assign(emb_iter, memberships_valid, metric=metric)

        key = f"normalize~{normalize}_dimThreshold~{dim_threshold}"

        # Place assignments back into full-size array; unembedded nodes get NaN
        full_cluster_ids = np.full(n_nodes, np.nan)
        full_cluster_ids[valid_indices] = cluster_ids
        results[key] = full_cluster_ids

np.savez(output_file, **results)
