# -*- coding: utf-8 -*-
import sys

import numpy as np
import pandas as pd
from scipy import sparse


def get_largest_component(func):
    """Decorator that extracts the largest connected component and
    ensures the adjacency matrix is symmetric and binary."""

    def wrapper(*args, **kwargs):
        net, community_labels = func(*args, **kwargs)
        _, component_ids = sparse.csgraph.connected_components(net, directed=False)
        largest_id = np.argmax(np.bincount(component_ids))
        keep = np.where(component_ids == largest_id)[0]
        net = net[keep, :][:, keep]
        net = net + net.T
        net.data = np.ones(net.nnz)
        # Re-index community labels to consecutive integers
        community_labels = np.unique(community_labels[keep], return_inverse=True)[1]
        return net, community_labels

    return wrapper


def graph_tool_to_sparse(g):
    """Convert a graph-tool Graph to a scipy sparse adjacency matrix."""
    edges = g.get_edges()
    n_nodes = g.num_vertices()
    return sparse.csr_matrix(
        (np.ones(g.num_edges()), (edges[:, 0], edges[:, 1])),
        shape=(n_nodes, n_nodes),
    )


def vertex_property_to_labels(g, prop_name):
    """Extract vertex property values and map them to consecutive integer labels."""
    raw_labels = [g.vp[prop_name][v] for v in g.vertices()]
    return np.unique(raw_labels, return_inverse=True)[1]


# --- Loader functions ---
# Each returns (adjacency_matrix, community_labels).
# The @get_largest_component decorator post-processes both outputs.


@get_largest_component
def load_airport():
    node_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/adv-net-sci-course/main/data/airport_network_v2/node_table.csv"
    )
    edge_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/adv-net-sci-course/main/data/airport_network_v2/edge_table.csv"
    )
    src, trg = edge_table["src"].values, edge_table["trg"].values
    n_nodes = len(node_table)
    adj = sparse.csr_matrix(
        (np.ones(len(src)), (src, trg)), shape=(n_nodes, n_nodes)
    )
    adj = adj + adj.T
    adj.data = np.ones(adj.nnz)
    return adj, node_table["region"].values


@get_largest_component
def load_polblog():
    edge_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/out.moreno_blogs_blogs"
    )
    node_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/ent.moreno_blogs_blogs.blog.orientation",
        sep=",",
        header=None,
        names=["class"],
    )
    # Edge indices in this dataset are 1-based
    src = edge_table.source.values - 1
    trg = edge_table.target.values - 1
    n_nodes = max(src.max(), trg.max()) + 1
    adj = sparse.csc_matrix(
        (np.ones(len(src)), (src, trg)), shape=(n_nodes, n_nodes)
    )
    return adj, node_table["class"].values


@get_largest_component
def load_football():
    import graph_tool.all as gt
    g = gt.collection.ns["football"]
    labels = np.array(g.vp["value"].get_array())
    adj = graph_tool_to_sparse(g)
    adj = adj + adj.T
    adj.data = np.ones(adj.nnz)
    return adj, labels


@get_largest_component
def load_highschool():
    import graph_tool.all as gt
    g = gt.collection.ns["sp_high_school_new/2011"]
    labels = vertex_property_to_labels(g, "class")
    adj = graph_tool_to_sparse(g)
    adj = adj + adj.T
    return adj, labels


@get_largest_component
def load_polbooks():
    import io, zipfile, urllib.request
    url = "https://netzschleuder.skewed.de/net/polbooks/files/polbooks.csv.zip"
    with urllib.request.urlopen(url) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
    edges = pd.read_csv(
        io.StringIO(z.read("edges.csv").decode()),
        comment="#",
        header=None,
        names=["source", "target"],
    )
    nodes = pd.read_csv(
        io.StringIO(z.read("nodes.csv").decode()),
        comment="#",
        header=None,
        names=["index", "label", "value", "_pos"],
    )
    src, trg = edges["source"].values, edges["target"].values
    n_nodes = len(nodes)
    adj = sparse.csr_matrix(
        (np.ones(len(src)), (src, trg)), shape=(n_nodes, n_nodes)
    )
    adj = adj + adj.T
    adj.data = np.ones(adj.nnz)
    labels = np.unique(nodes["value"].values, return_inverse=True)[1]
    return adj, labels


@get_largest_component
def load_cora():
    edge_table = pd.read_csv(
        "https://raw.githubusercontent.com/sk-classroom/GraphNeuralNetworks/master/derived/preprocess/cora.cites",
        sep="\t",
        header=None,
        names=["src", "trg"],
    )
    node_table = pd.read_csv(
        "https://raw.githubusercontent.com/sk-classroom/GraphNeuralNetworks/master/derived/preprocess/cora.content",
        sep="\t",
        header=None,
    )
    # Keep only node_id (first col) and field label (last col)
    node_table = (
        node_table.iloc[:, [0, -1]]
        .rename(columns={0: "node_id", 1434: "field"})
        .sort_values(by="node_id")
    )
    unique_ids, new_ids = np.unique(node_table["node_id"].values, return_inverse=True)
    id_mapping = dict(zip(unique_ids, new_ids))
    edge_table["src"] = edge_table["src"].map(id_mapping)
    edge_table["trg"] = edge_table["trg"].map(id_mapping)
    node_table["node_id"] = new_ids

    n_nodes = len(node_table)
    adj = sparse.csr_matrix(
        (np.ones(len(edge_table)), (edge_table["src"], edge_table["trg"])),
        shape=(n_nodes, n_nodes),
    )
    adj = adj + adj.T
    adj.data = np.ones(adj.nnz)
    return adj, node_table["field"].values


@get_largest_component
def load_karate():
    import graph_tool.all as gt
    g = gt.collection.ns["karate/77"]
    labels = vertex_property_to_labels(g, "groups")
    adj = graph_tool_to_sparse(g)
    return adj, labels


# Maps dataset name (used in Snakemake wildcards) to its loader function
NETWORK_LOADERS = {
    "polblog": load_polblog,
    "cora": load_cora,
    "airport": load_airport,
    "football": load_football,
    "highschool": load_highschool,
    "polbooks": load_polbooks,
    "karate": load_karate,
}

if "snakemake" in sys.modules:
    params = snakemake.params["parameters"]
    output_file = snakemake.output["output_file"]
    output_node_file = snakemake.output["output_node_file"]
else:
    params = {"netdata": "polblog"}
    output_file = ""
    output_node_file = ""

data_name = params["netdata"]
net, labels = NETWORK_LOADERS[data_name]()

# Sanity checks
n_components, _ = sparse.csgraph.connected_components(net, directed=False)
assert n_components == 1, "The network must have exactly one connected component."
assert net.shape[0] == len(labels)
assert (net != net.T).nnz == 0, "The network must be symmetric."

sparse.save_npz(output_file, net)
node_table = pd.DataFrame({"node_id": np.arange(len(labels)), "membership": labels})
node_table.to_csv(output_node_file, index=False)
