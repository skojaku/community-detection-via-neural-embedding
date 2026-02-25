"""Generate networks using a planted-partition (stochastic block) model."""
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
    """Generate a planted-partition model network using graph-tool's SBM generator.

    Nodes are assigned to communities in round-robin order, then an SBM is
    sampled with in-community and cross-community edge probabilities set by
    the desired average degree and mixing rate.
    """
    import graph_tool.all as gt
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

    # Scale by community sizes to get expected edge counts between blocks
    edge_count_matrix = np.diag(community_sizes) @ block_probs @ np.diag(community_sizes)

    g = gt.generate_sbm(
        b=memberships,
        probs=edge_count_matrix,
        micro_degs=False,
        in_degs=np.full(n_nodes, avg_degree),
        out_degs=np.full(n_nodes, avg_degree),
    )

    adj_matrix = gt.adjacency(g).T
    adj_matrix.data = np.ones_like(adj_matrix.data)  # binarize edge weights

    return adj_matrix, memberships


# Generate and save
adj_matrix, memberships = generate_network(n_nodes, n_communities, avg_degree, mixing_rate)

sparse.save_npz(output_file, adj_matrix)
pd.DataFrame({
    "node_id": np.arange(adj_matrix.shape[0], dtype=int),
    "membership": memberships,
}).to_csv(output_node_file, index=False)
