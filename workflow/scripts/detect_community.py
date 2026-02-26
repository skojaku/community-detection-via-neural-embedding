"""Community detection algorithms.

Supported models:
- "infomap": Infomap
- "flatsbm": Degree-corrected SBM
- "bp": Belief propagation
"""
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

# ============================================================
# Snakemake / standalone parameter loading
# ============================================================
if "snakemake" in sys.modules:
    netfile = snakemake.input["net_file"]
    com_file = snakemake.input["com_file"]
    params = snakemake.params["parameters"]
    model_name = params["model_name"]
    output_file = snakemake.output["output_file"]
else:
    netfile = "../../data/multi_partition_model/networks/net_n~10000_K~50_cave~50_mu~0.30_sample~0.npz"
    com_file = "../../data/multi_partition_model/networks/node_n~10000_K~50_cave~50_mu~0.30_sample~0.npz"
    model_name = "bp"
    output_file = "../data/"


# ============================================================
# Helpers
# ============================================================
def _contiguous_labels(labels):
    """Remap arbitrary integer labels to contiguous 0-indexed IDs."""
    return np.unique(labels, return_inverse=True)[1]


# ============================================================
# Community detection algorithms
# ============================================================
def detect_by_infomap(adj, num_communities):
    """Detect communities using Infomap."""
    import infomap

    rows, cols, _ = sparse.find(adj + adj.T)
    im = infomap.Infomap("--two-level --directed")
    for i in range(len(rows)):
        im.add_link(rows[i], cols[i], 1)
    im.run()

    community_ids = np.zeros(adj.shape[0])
    for node in im.tree:
        if node.is_leaf:
            community_ids[node.node_id] = node.module_id
    return _contiguous_labels(community_ids)


def detect_by_flatsbm(adj, num_communities):
    """Detect communities using a degree-corrected flat SBM (graph-tool)."""
    import graph_tool.all as gt

    rows, cols, _ = sparse.find(adj)
    graph = gt.Graph(directed=False)
    graph.add_edge_list(np.vstack([rows, cols]).T)
    state = gt.minimize_blockmodel_dl(
        graph,
        state_args={"B_min": num_communities, "B_max": num_communities},
        multilevel_mcmc_args={"B_max": num_communities, "B_min": num_communities},
    )
    blocks = state.get_blocks()
    return _contiguous_labels(np.array(blocks.a))


def detect_by_belief_propagation(adj, num_communities, memberships):
    """Detect communities using belief propagation."""
    import belief_propagation as bp

    return bp.detect(adj, q=int(num_communities), init_memberships=memberships)


# ============================================================
# Load data
# ============================================================
net = sparse.load_npz(netfile)
memberships = pd.read_csv(com_file)["membership"].values.astype(int)
num_communities = len(set(memberships))

# ============================================================
# Extract the largest connected component
# ============================================================
net = sparse.csr_matrix(net)
_, component_labels = connected_components(net)
unique_labels, counts = np.unique(component_labels, return_counts=True)
largest_component_label = unique_labels[np.argmax(counts)]
lcc_node_ids = np.where(component_labels == largest_component_label)[0]

num_lcc_nodes = len(lcc_node_ids)
projection = sparse.csr_matrix(
    (np.ones(num_lcc_nodes), (lcc_node_ids, np.arange(num_lcc_nodes))),
    shape=(net.shape[0], num_lcc_nodes),
)
lcc_net = projection.T @ net @ projection

# ============================================================
# Run community detection on the LCC
# ============================================================
if model_name == "infomap":
    group_ids = detect_by_infomap(lcc_net, num_communities)
elif model_name == "flatsbm":
    group_ids = detect_by_flatsbm(lcc_net, num_communities)
elif model_name == "bp":
    group_ids = detect_by_belief_propagation(
        lcc_net, num_communities, memberships[lcc_node_ids]
    )

# Embed LCC results back into full-graph array (NaN for non-LCC nodes)
all_group_ids = np.full(net.shape[0], np.nan)
all_group_ids[lcc_node_ids] = group_ids

# ============================================================
# Save
# ============================================================
np.savez(output_file, group_ids=all_group_ids)
