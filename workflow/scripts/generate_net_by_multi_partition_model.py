"""Generate networks using a planted-partition (stochastic block) model."""
import sys

import igraph as ig
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
    """Generate a planted-partition model network using igraph's SBM.

    Nodes are assigned to communities in round-robin order, then an SBM is
    sampled with in-community and cross-community edge probabilities set by
    the desired average degree and mixing rate.
    """
    memberships = np.sort(np.arange(n_nodes) % n_communities)
    community_sizes = np.bincount(memberships, minlength=n_communities).tolist()

    # Derive in-community and cross-community connection probabilities
    avg_degree_out = np.maximum(1, mixing_rate * avg_degree)
    avg_degree_in = n_communities * avg_degree - (n_communities - 1) * avg_degree_out
    prob_out = avg_degree_out / n_nodes
    prob_in = avg_degree_in / n_nodes

    block_probs = np.full((n_communities, n_communities), prob_out)
    np.fill_diagonal(block_probs, prob_in)

    g = ig.Graph.SBM(
        n=n_nodes,
        pref_matrix=block_probs.tolist(),
        block_sizes=community_sizes,
        directed=False,
        loops=False,
    )

    edges = np.array(g.get_edgelist(), dtype=np.int32)
    if len(edges) == 0:
        adj_matrix = sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    else:
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(len(rows), dtype=np.float32)
        adj_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    return adj_matrix, memberships


# Generate and save
adj_matrix, memberships = generate_network(n_nodes, n_communities, avg_degree, mixing_rate)

sparse.save_npz(output_file, adj_matrix)
pd.DataFrame({
    "node_id": np.arange(adj_matrix.shape[0], dtype=int),
    "membership": memberships,
}).to_csv(output_node_file, index=False)
