"""Graph embedding pipeline.

Loads a network, selects a model from the registry, computes embeddings
on the largest connected component, and saves the result.
"""
import logging
import sys

# Re-exec with the correct Python environment if embcom is not available.
try:
    import embcom as _embcom_check  # noqa: F401
    del _embcom_check
except ModuleNotFoundError:
    import os

    base = os.path.expanduser("~/miniforge3/envs")
    neuralemb_py = os.path.join(base, "neuralemb", "bin", "python3")
    if sys.executable != neuralemb_py and os.path.isfile(neuralemb_py):
        os.execv(neuralemb_py, [neuralemb_py] + sys.argv)

import GPUtil
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

import embcom

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# =====================
# Input
# =====================
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    com_file = snakemake.input["com_file"]
    embfile = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    dim = int(params["dim"])
    window_length = int(params["window_length"])
    model_name = params["model_name"]
    num_walks = int(params["nWalks"]) if "nWalks" in params else 40
else:
    netfile = "../../data/empirical/networks/net_netdata~polblog.npz"
    com_file = "../../data/empirical/networks/node_netdata~polblog.npz"
    embfile = "tmp.npz"
    dim = 64
    window_length = 10
    model_name = "leigenmap"
    num_walks = 40

# =====================
# Load network
# =====================
net = sparse.load_npz(netfile)
net = net + net.T
net.data[:] = 1  # Binarize edge weights
n_nodes = net.shape[0]

true_membership = pd.read_csv(com_file)["membership"].values.astype(int)
if dim == 0:
    dim = len(set(true_membership)) - 1
    dim = min(n_nodes - 1, dim)

# =====================
# GPU device selection
# =====================
if "touch" in model_name:
    gpu_id = GPUtil.getFirstAvailable(
        order="random",
        maxLoad=1,
        maxMemory=0.3,
        attempts=99999,
        interval=60,
        verbose=False,
    )[0]
    device = f"cuda:{gpu_id}"
else:
    device = "cpu"

# =====================
# Model instantiation
# =====================
WALK_PARAMS = dict(window_length=window_length, num_walks=num_walks)
TORCH_PARAMS = dict(
    window=window_length,
    num_walks=num_walks,
    vector_size=dim,
    batch_size=256,
    device=device,
    negative=1,
)

MODEL_REGISTRY = {
    "node2vec": (embcom.embeddings.Node2Vec, WALK_PARAMS),
    "depthfirst-node2vec": (
        embcom.embeddings.Node2Vec,
        {**WALK_PARAMS, "p": 100, "q": 1},
    ),
    "deepwalk": (embcom.embeddings.DeepWalk, WALK_PARAMS),
    "line": (
        embcom.embeddings.Node2Vec,
        {"window_length": 1, "num_walks": num_walks * 10, "p": 1, "q": 1},
    ),
    "leigenmap": (embcom.embeddings.LaplacianEigenMap, {}),
    "adjspec": (embcom.embeddings.AdjacencySpectralEmbedding, {}),
    "modspec": (embcom.embeddings.ModularitySpectralEmbedding, {}),
    "nonbacktracking": (
        embcom.embeddings.NonBacktrackingSpectralEmbedding,
        {},
    ),
    "node2vec-matrixfact": (
        embcom.embeddings.Node2VecMatrixFactorization,
        {"window_length": window_length, "blocking_membership": None},
    ),
    "linearized-node2vec": (
        embcom.embeddings.LinearizedNode2Vec,
        {"window_length": window_length},
    ),
}

TORCH_MODELS = {"torch-node2vec", "torch-modularity", "torch-laplacian-eigenmap"}
if model_name in TORCH_MODELS:
    import node2vecs

    torch_model_map = {
        "torch-node2vec": node2vecs.TorchNode2Vec,
        "torch-modularity": node2vecs.TorchModularity,
        "torch-laplacian-eigenmap": node2vecs.TorchLaplacianEigenMap,
    }
    model = torch_model_map[model_name](**TORCH_PARAMS)
else:
    model_cls, model_kwargs = MODEL_REGISTRY[model_name]
    model = model_cls(**model_kwargs)

# =====================
# Embedding
# =====================
net = sparse.csr_matrix(net)
component_labels = connected_components(net)[1]
unique_labels, counts = np.unique(component_labels, return_counts=True)
largest_component_label = unique_labels[np.argmax(counts)]
lcc_node_ids = np.where(component_labels == largest_component_label)[0]

n_lcc = len(lcc_node_ids)
projection = sparse.csr_matrix(
    (np.ones(n_lcc), (lcc_node_ids, np.arange(n_lcc))),
    shape=(n_nodes, n_lcc),
)
lcc_net = projection.T @ net @ projection

model.fit(lcc_net)

if model_name in TORCH_MODELS:
    lcc_emb = model.transform()
else:
    # TruncatedSVD needs at least 2 features; leigenmap uses dim+1 components internally,
    # so we need lcc_net.shape[0] >= effective_dim + 2.
    effective_dim = min(dim, lcc_net.shape[0] - 2)
    if effective_dim < 1:
        lcc_emb = np.zeros((n_lcc, 1))
    else:
        lcc_emb = model.transform(dim=effective_dim)

non_lcc_node_ids = np.where(component_labels != largest_component_label)[0]
emb = projection @ lcc_emb
emb[non_lcc_node_ids, :] = np.nan

# =====================
# Save
# =====================
np.savez_compressed(
    embfile,
    emb=emb,
    window_length=window_length,
    dim=dim,
    model_name=model_name,
)
