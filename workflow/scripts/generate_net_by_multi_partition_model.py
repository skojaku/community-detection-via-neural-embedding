"""Generate networks using a planted-partition (stochastic block) model.

Uses a pure numpy/scipy implementation; no graph_tool dependency required.
"""
import sys

import numpy as np
import pandas as pd
from scipy import sparse


if "snakemake" in sys.modules:
    params = snakemake.params["parameters"]
    n_nodes = int(params["n"])
    n_communities = int(params["q"])
    avg_degree = int(params["cave"])
    mixing_rate = float(params["mu"])
    output_file = snakemake.output["output_file"]
    output_node_file = snakemake.output["output_node_file"]
else:
    n_nodes = 10000
    n_communities = 64
    avg_degree = 10
    mixing_rate = 0.4
    output_file = ""
    output_node_file = ""


def generate_network(n_nodes, n_communities, avg_degree, mixing_rate):
    """Generate a planted-partition model network using a pure numpy/scipy SBM.

    Nodes are assigned to communities in round-robin order, then an SBM is
    sampled with in-community and cross-community edge probabilities set by
    the desired average degree and mixing rate.
    """
    memberships = np.sort(np.arange(n_nodes) % n_communities)

    # Compute community sizes
    community_sizes = np.bincount(memberships, minlength=n_communities).astype(float)

    # Derive in-community and cross-community connection probabilities
    avg_degree_out = np.maximum(1, mixing_rate * avg_degree)
    avg_degree_in = n_communities * avg_degree - (n_communities - 1) * avg_degree_out
    prob_out = avg_degree_out / n_nodes
    prob_in = avg_degree_in / n_nodes

    # Block probability matrix: prob_out everywhere, boosted to prob_in on diagonal
    block_probs = np.full((n_communities, n_communities), prob_out)
    np.fill_diagonal(block_probs, prob_in)

    # Sample edges for each pair of communities
    rows, cols = [], []
    for r in range(n_communities):
        nodes_r = np.where(memberships == r)[0]
        for c in range(r, n_communities):
            nodes_c = np.where(memberships == c)[0]
            p = block_probs[r, c]
            if r == c:
                # Upper triangle only to avoid duplicates, then symmetrize
                n_r = len(nodes_r)
                pairs = n_r * (n_r - 1) // 2
                edges = np.random.binomial(pairs, p)
                if edges > 0:
                    idx = np.random.choice(pairs, size=edges, replace=False)
                    tri_r, tri_c = np.triu_indices(n_r, k=1)
                    src = nodes_r[tri_r[idx]]
                    dst = nodes_c[tri_c[idx]]
                    rows.extend(np.concatenate([src, dst]))
                    cols.extend(np.concatenate([dst, src]))
            else:
                pairs = len(nodes_r) * len(nodes_c)
                edges = np.random.binomial(pairs, p)
                if edges > 0:
                    idx = np.random.choice(pairs, size=edges, replace=False)
                    src = nodes_r[idx // len(nodes_c)]
                    dst = nodes_c[idx % len(nodes_c)]
                    rows.extend(np.concatenate([src, dst]))
                    cols.extend(np.concatenate([dst, src]))

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    adj_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    # Binarize (remove duplicate edges)
    adj_matrix.data = np.ones_like(adj_matrix.data)

    return adj_matrix, memberships


# Generate and save
adj_matrix, memberships = generate_network(n_nodes, n_communities, avg_degree, mixing_rate)

sparse.save_npz(output_file, adj_matrix)
pd.DataFrame({
    "node_id": np.arange(adj_matrix.shape[0], dtype=int),
    "membership": memberships,
}).to_csv(output_node_file, index=False)
