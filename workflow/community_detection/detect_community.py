# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-03 22:30:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-11 21:34:22
"""Community detection algorithms.

- "infomap": Infomap
- "flatsbm": Degree-corrected SBM
- "nestedsbm": Degree-corrected SBM with nested structure (not implemented)
- "bp": Belief propagation
"""
# %%
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import belief_propagation as bp

if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    com_file = snakemake.input["com_file"]
    params = snakemake.params["parameters"]
    model_name = params["model_name"]
    #mu = params["mu"] if "mu" in params["mu"] else None
    #ave_deg = params["cave"] if "cave" in params["cave"] else None
    output_file = snakemake.output["output_file"]
else:
    netfile = "../../data/multi_partition_model/networks/net_n~10000_K~50_cave~50_mu~0.30_sample~0.npz"
    com_file = "../../data/multi_partition_model/networks/node_n~10000_K~50_cave~50_mu~0.30_sample~0.npz"
    model_name = "bp"

    output_file = "../data/"

#
# Load
#
net = sparse.load_npz(netfile)

memberships = pd.read_csv(com_file)["membership"].values.astype(int)
K = len(set(memberships))


# %%
#
# Communiyt detection
#
def detect_by_infomap(A, K):
    import infomap
    r, c, v = sparse.find(A + A.T)
    im = infomap.Infomap("--two-level --directed")
    for i in range(len(r)):
        im.add_link(r[i], c[i], 1)
    im.run()
    cids = np.zeros(A.shape[0])
    for node in im.tree:
        if node.is_leaf:
            cids[node.node_id] = node.module_id
    return np.unique(cids, return_inverse=True)[1]


def detect_by_flatsbm(A, K):
    import graph_tool.all as gt
    r, c, v = sparse.find(A)
    g = gt.Graph(directed=False)
    g.add_edge_list(np.vstack([r, c]).T)
    state = gt.minimize_blockmodel_dl(
        g,
        state_args={"B_min": K, "B_max": K},
        multilevel_mcmc_args={"B_max": K, "B_min": K},
    )
    b = state.get_blocks()
    return np.unique(np.array(b.a), return_inverse=True)[1]

def detect_by_belief_propagation(A, K, memberships):
    return bp.detect(A, q=int(K), init_memberships = memberships)


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

if model_name == "infomap":
    group_ids = detect_by_infomap(net_, K)
elif model_name == "flatsbm":
    group_ids = detect_by_flatsbm(net_, K)
elif model_name == "bp":
    group_ids = detect_by_belief_propagation(net_, K, memberships[ids])

n_nodes = net.shape[0]
group_ids_ = np.zeros(n_nodes) * np.nan
group_ids_[ids] = group_ids
print(group_ids)
# %%
# Save
#
np.savez(output_file, group_ids=group_ids_)

# %%
