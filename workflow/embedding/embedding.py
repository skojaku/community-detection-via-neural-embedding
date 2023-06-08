# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 15:08:01
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-08 16:33:41
# %%
import logging
import sys

import GPUtil
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

import embcom

# import node2vecs

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

#
# Input
#
if "snakemake" in sys.modules:
    #    input_file = snakemake.input['input_file']
    #    output_file = snakemake.output['output_file']
    netfile = snakemake.input["net_file"]
    com_file = snakemake.input["com_file"]
    embfile = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    dim = int(params["dim"])
    window_length = int(params["window_length"])
    model_name = params["model_name"]
    num_walks = int(params["nWalks"]) if "nWalks" in params else 40
else:
    netfile = "../../data/multi_partition_model/networks/net_n~100000_K~2_cave~10_mu~0.10_sample~0.npz"
    com_file = "../../data/multi_partition_model/networks/node_n~100000_K~2_cave~10_mu~0.10_sample~0.npz"
    embfile = "tmp.npz"
    dim = 64
    window_length = 10
    model_name = "torch-modularity"
    num_walks = 40


net = sparse.load_npz(netfile)
net = net + net.T
net.data = net.data * 0 + 1

true_membership = pd.read_csv(com_file)["membership"].values.astype(int)

if dim == 0:
    dim = len(set(true_membership)) - 1
    dim = np.minimum(net.shape[0] - 1, dim)

if "touch" in model_name:
    device = GPUtil.getFirstAvailable(
        order="random",
        maxLoad=1,
        maxMemory=0.3,
        attempts=99999,
        interval=60 * 1,
        verbose=False,
    )[0]
    device = f"cuda:{device}"
else:
    device = "cpu"

#
# Embedding models
#
if model_name == "levy-word2vec":
    model = embcom.embeddings.LevyWord2Vec(
        window_length=window_length, num_walks=num_walks
    )
elif model_name == "node2vec":
    # model = fastnode2vec.Node2Vec(window_length=window_length, num_walks=num_walks)
    model = embcom.embeddings.Node2Vec(window_length=window_length, num_walks=num_walks)
elif model_name == "depthfirst-node2vec":
    # model = fastnode2vec.Node2Vec(
    #    window_length=window_length, num_walks=num_walks, p=10, q=0.1
    # )
    model = embcom.embeddings.Node2Vec(
        window_length=window_length, num_walks=num_walks, p=100, q=1
    )
elif model_name == "deepwalk":
    # model = fastnode2vec.DeepWalk(window_length=window_length, num_walks=num_walks)
    model = embcom.embeddings.DeepWalk(
        window_length=window_length, num_walks=num_walks * 3
    )
elif model_name == "line":
    model = embcom.embeddings.Node2Vec(
        window_length=1, num_walks=num_walks * 10, p=1, q=1
    )
elif model_name == "glove":
    model = embcom.embeddings.Glove(window_length=window_length, num_walks=num_walks)
elif model_name == "leigenmap":
    model = embcom.embeddings.LaplacianEigenMap()
elif model_name == "adjspec":
    model = embcom.embeddings.AdjacencySpectralEmbedding()
elif model_name == "modspec":
    model = embcom.embeddings.ModularitySpectralEmbedding()
elif model_name == "nonbacktracking":
    model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
elif model_name == "node2vec-matrixfact":
    model = embcom.embeddings.Node2VecMatrixFactorization(
        window_length=window_length, blocking_membership=None
    )
elif model_name == "highorder-modspec":
    model = embcom.embeddings.HighOrderModularitySpectralEmbedding(
        window_length=window_length
    )
elif model_name == "linearized-node2vec":
    model = embcom.embeddings.LinearizedNode2Vec(window_length=window_length)
elif model_name == "non-backtracking-node2vec":
    model = embcom.embeddings.NonBacktrackingNode2Vec(
        window_length=window_length, num_walks=num_walks
    )
elif model_name == "non-backtracking-deepwalk":
    model = embcom.embeddings.NonBacktrackingDeepWalk(
        window_length=window_length, num_walks=num_walks
    )
elif model_name == "non-backtracking-glove":
    model = embcom.embeddings.NonBacktrackingGlove(
        window_length=window_length, num_walks=num_walks
    )
elif model_name == "torch-node2vec":
    model = node2vecs.TorchNode2Vec(
        window=window_length,
        num_walks=num_walks,
        vector_size=dim,
        batch_size=256,
        device=device,
        negative=1,
    )
elif model_name == "torch-modularity":
    model = node2vecs.TorchModularity(
        window=window_length,
        num_walks=num_walks,
        vector_size=dim,
        batch_size=256,
        device=device,
        negative=1,
    )
elif model_name == "torch-laplacian-eigenmap":
    model = node2vecs.TorchLaplacianEigenMap(
        window=window_length,
        num_walks=num_walks,
        vector_size=dim,
        batch_size=256,
        device=device,
        negative=1,
    )

# %%
# Embedding
#

# Get the largest connected component
net = sparse.csr_matrix(net)
component_ids = connected_components(net)[1]
u_component_ids, freq = np.unique(component_ids, return_counts=True)
ids = np.where(u_component_ids[np.argmax(freq)] == component_ids)[0]
H = sparse.csr_matrix(
    (np.ones_like(ids), (ids, np.arange(len(ids)))), shape=(net.shape[0], len(ids))
)
HT = sparse.csr_matrix(H.T)
net_ = HT @ net @ H
model.fit(net_)

if model_name in ["torch-node2vec", "torch-modularity", "torch-laplacian-eigenmap"]:
    emb_ = model.transform()
else:
    emb_ = model.transform(dim=dim)

# Enlarge the embedding to the size of the original net
# All nodes that do not belong to the largest connected component have nan
ids = np.where(u_component_ids[np.argmax(freq)] != component_ids)[0]
emb = H @ emb_
emb[ids, :] = np.nan
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
labels = np.unique([d[1]['club'] for d in G.nodes(data=True)], return_inverse=True)[1]
# %%
#
# Save
#
np.savez_compressed(
    embfile,
    emb=emb,
    window_length=window_length,
    dim=dim,
    model_name=model_name,
)
