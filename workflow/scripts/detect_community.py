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


def _get_conda_python_bin():
    """Resolve the Python binary inside the neuralemb conda environment.

    Used by subprocess fallbacks when an optional dependency (infomap,
    graph_tool) is not importable in the current interpreter.
    """
    import os

    conda_prefix = os.environ.get(
        "CONDA_PREFIX", os.path.expanduser("~/miniforge3/envs/neuralemb")
    )
    if "envs" not in conda_prefix:
        conda_prefix = os.path.join(conda_prefix, "envs", "neuralemb")
    return os.path.join(conda_prefix, "bin", "python3")


# ============================================================
# Community detection algorithms
# ============================================================
def detect_by_infomap(adj, num_communities):
    """Detect communities using Infomap."""
    try:
        import infomap
    except ModuleNotFoundError:
        return _detect_by_infomap_subprocess(adj, num_communities)

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


def _detect_by_infomap_subprocess(adj, num_communities):
    """Run Infomap via subprocess when the infomap package is unavailable."""
    import os
    import subprocess
    import tempfile

    python_bin = _get_conda_python_bin()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.npz")
        output_path = os.path.join(tmpdir, "output.npy")
        sparse.save_npz(input_path, sparse.csr_matrix(adj))

        script = f"""
import numpy as np
from scipy import sparse
import infomap

A = sparse.load_npz("{input_path}")
r, c, v = sparse.find(A + A.T)
im = infomap.Infomap("--two-level --directed")
for i in range(len(r)):
    im.add_link(r[i], c[i], 1)
im.run()
cids = np.zeros(A.shape[0])
for node in im.tree:
    if node.is_leaf:
        cids[node.node_id] = node.module_id
result = np.unique(cids, return_inverse=True)[1]
np.save("{output_path}", result)
"""
        result = subprocess.run(
            [python_bin, "-c", script],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"infomap subprocess failed:\n{result.stderr}")
        return np.load(output_path)


def detect_by_flatsbm(adj, num_communities):
    """Detect communities using a degree-corrected flat SBM (graph-tool)."""
    try:
        import graph_tool.all as gt
    except ModuleNotFoundError:
        return _detect_by_flatsbm_subprocess(adj, num_communities)

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


def _detect_by_flatsbm_subprocess(adj, num_communities):
    """Run flat SBM via subprocess when graph_tool is unavailable."""
    import os
    import subprocess
    import tempfile

    python_bin = _get_conda_python_bin()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.npz")
        output_path = os.path.join(tmpdir, "output.npy")
        sparse.save_npz(input_path, sparse.csr_matrix(adj))

        script = f"""
import numpy as np
from scipy import sparse
import graph_tool.all as gt

A = sparse.load_npz("{input_path}")
K = {num_communities}
r, c, v = sparse.find(A)
g = gt.Graph(directed=False)
g.add_edge_list(np.vstack([r, c]).T)
state = gt.minimize_blockmodel_dl(
    g,
    state_args={{"B_min": K, "B_max": K}},
    multilevel_mcmc_args={{"B_max": K, "B_min": K}},
)
b = state.get_blocks()
result = np.unique(np.array(b.a), return_inverse=True)[1]
np.save("{output_path}", result)
"""
        result = subprocess.run(
            [python_bin, "-c", script],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"flatsbm subprocess failed:\n{result.stderr}")
        return np.load(output_path)


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
