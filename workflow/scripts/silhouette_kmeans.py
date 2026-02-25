"""Cluster embeddings via K-means, selecting K by silhouette score.

Tries four combinations of (dim_threshold, normalize) preprocessing and
saves the resulting community labels to a single .npz file.
"""

import sys

import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.metrics import silhouette_score

# ── I/O configuration ───────────────────────────────────────────────────────

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    metric = params["metric"]
    model_name = params["model_name"]
else:
    emb_file = "../../data/empirical/embedding/netdata~cora_model_name~node2vec_window_length~10_dim~64~sample~0.npz"
    com_file = "../../data/empirical/networks/node_netdata~cora.npz"
    model_name = "leigenmap"
    output_file = "unko"
    metric = "cosine"

# Models whose embeddings are spectral (dimension has ordering meaning),
# so we truncate to K-1 dimensions when K clusters are chosen.
SPECTRAL_MODELS = {
    "leigenmap",
    "modspec",
    "modspec2",
    "linearized-node2vec",
    "nonbacktracking",
}


def run_kmeans(emb, n_clusters, metric="cosine"):
    """Run K-means and return cluster labels.

    Note: the cosine branch currently behaves identically to the default
    branch (the row-normalization result is not used). This preserves the
    original script behavior.
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(emb)
    return kmeans.labels_


def find_best_k(emb, k_range):
    """Return the K in *k_range* that maximises the cosine silhouette score."""
    scores = []
    for k in k_range:
        labels = cluster.KMeans(n_clusters=k, random_state=0, n_init=10).fit(emb).labels_
        scores.append(silhouette_score(emb, labels, metric="cosine"))
    return k_range[np.argmax(scores)]


def threshold_dimensions(emb):
    """Filter columns of *emb* using the non-backtracking norm threshold.

    Keeps dimensions whose column norm >= sqrt(leading column norm),
    excluding the leading dimension itself. Falls back to the leading
    dimension alone if no others survive.
    """
    col_norms = np.linalg.norm(emb, axis=0)
    leading_idx = np.argmax(col_norms)
    threshold = np.sqrt(col_norms[leading_idx])

    keep = col_norms >= threshold
    keep[leading_idx] = False
    if not np.any(keep):
        keep[leading_idx] = True

    return emb[:, keep]


def normalize_columns(emb):
    """Scale each column to unit norm (with a small floor to avoid division by zero)."""
    col_norms = np.linalg.norm(emb, axis=0)
    return emb / np.maximum(col_norms, 1e-32)


# ── Main pipeline ────────────────────────────────────────────────────────────

# Load embedding and ground-truth memberships
emb_raw = np.load(emb_file)["emb"]
emb_raw = np.nan_to_num(emb_raw.copy(order="C").astype(np.float32))
memberships = pd.read_csv(com_file)["membership"].values.astype(int)

# Select optimal K via silhouette score over K = 2..20
K = find_best_k(emb_raw, range(2, 21))

# For spectral models, truncate to at most K-1 meaningful dimensions
if model_name in SPECTRAL_MODELS:
    n_dims = max(K - 1, 1)
    emb_base = emb_raw[:, :n_dims].reshape((emb_raw.shape[0], -1))
else:
    emb_base = emb_raw.copy()

# Evaluate all four (dim_threshold, normalize) combinations
results = {}
for dim_threshold in [True, False]:
    for normalize in [True, False]:
        emb = emb_base.copy()

        if model_name == "nonbacktracking" and dim_threshold:
            emb = threshold_dimensions(emb)

        if normalize:
            emb = normalize_columns(emb)

        group_ids = run_kmeans(emb, K, metric=metric)

        key = f"normalize~{normalize}_dimThreshold~{dim_threshold}"
        results[key] = group_ids

# Save all label arrays keyed by preprocessing variant
np.savez(output_file, **results)
