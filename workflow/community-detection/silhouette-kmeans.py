"""Evaluate the detected communities using the element-centric similarity."""

# %%
import sys
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import normalized_mutual_info_score

if "snakemake" in sys.modules:
    emb_file = snakemake.input["emb_file"]
    com_file = snakemake.input["com_file"]
    net_file = snakemake.input["net_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    metric = params["metric"]
    model_name = params["model_name"]


else:
    emb_file = "../../data/empirical/embedding/netdata~cora_model_name~node2vec_window_length~10_dim~64~sample~0.npz"
    com_file = "../../data/empirical/networks/node_netdata~cora.npz"
    net_file = "../../data/empirical/networks/net_netdata~cora.npz"
    model_name = "leigenmap"
    output_file = "unko"
    metric = "cosine"


from sklearn import cluster


def KMeans(emb, K, metric="cosine"):
    if (metric == "cosine") & (emb.shape[1] > 1):
        X = np.einsum(
            "ij,i->ij", emb, 1 / np.maximum(np.linalg.norm(emb, axis=1), 1e-24)
        )
        X = emb
    else:
        X = emb.copy()
    kmeans = cluster.KMeans(n_clusters=K, random_state=0).fit(X)
    # kmeans = cluster.MiniBatchKMeans(n_clusters=K, random_state=0).fit(X)
    # kmeans = cluster.MiniBatchKMeans(n_clusters=K, random_state=0).fit(X)
    return kmeans.labels_


# Load emebdding
emb = np.load(emb_file)["emb"]
emb = emb.copy(order="C").astype(np.float32)
memberships = pd.read_csv(com_file)["membership"].values.astype(int)

emb = np.nan_to_num(emb)


A = sparse.load_npz(net_file)

# Determine the optimal number of clusters K using silhouette score
silhouette_scores = []
K_range = range(2, 21)  # Testing K from 2 to 10
for k in K_range:
    kmeans_test = cluster.KMeans(n_clusters=k, random_state=0, n_init=10).fit(emb)
    score = silhouette_score(emb, kmeans_test.labels_, metric="cosine")
    silhouette_scores.append(score)

# Select the K with the highest silhouette score
K = K_range[np.argmax(silhouette_scores)]

if model_name in [
    "leigenmap",
    "modspec",
    "modspec2",
    "linearized-node2vec",
    "nonbacktracking",
]:
    emb_copy = emb.copy()[:, : np.maximum((K - 1), 1)].reshape((emb.shape[0], -1))
else:
    emb_copy = emb.copy()
# %%
# Normalize the eigenvector by dimensions
results = {}
for dimThreshold in [True, False]:
    for normalize in [True, False]:
        emb = emb_copy.copy()
        if (model_name == "nonbacktracking") & dimThreshold:
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
        group_ids = KMeans(emb, K, metric=metric)

        key = f"normalize~{normalize}_dimThreshold~{dimThreshold}"
        results[key] = group_ids
        print(normalized_mutual_info_score(group_ids, memberships))
# %%
K, len(set(memberships))
# %%
# Save
#
np.savez(output_file, **results)

# %%
