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
    import io, zipfile, urllib.request
    url = "https://networks.skewed.de/net/football/files/football.csv.zip"
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
def load_highschool():
    import io, zipfile, urllib.request
    url = "https://networks.skewed.de/net/sp_high_school_new/files/2011.csv.zip"
    with urllib.request.urlopen(url) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
    # The contact list: columns t, i, j, Ci, Cj
    contacts = pd.read_csv(
        io.StringIO(z.read("edges.csv").decode()),
        comment="#",
        header=None,
        names=["t", "i", "j", "Ci", "Cj"],
        sep=r"\s+",
    )
    # Build node-to-class mapping from contact data
    node_class = pd.concat(
        [contacts[["i", "Ci"]].rename(columns={"i": "node", "Ci": "class"}),
         contacts[["j", "Cj"]].rename(columns={"j": "node", "Cj": "class"})]
    ).drop_duplicates("node").sort_values("node").reset_index(drop=True)
    unique_nodes, new_ids = np.unique(node_class["node"].values, return_inverse=True)
    id_map = dict(zip(unique_nodes, range(len(unique_nodes))))
    n_nodes = len(unique_nodes)
    src = contacts["i"].map(id_map).values
    trg = contacts["j"].map(id_map).values
    adj = sparse.csr_matrix(
        (np.ones(len(src)), (src, trg)), shape=(n_nodes, n_nodes)
    )
    adj = adj + adj.T
    adj.data = np.ones(adj.nnz)
    labels = np.unique(node_class["class"].values, return_inverse=True)[1]
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
    import io, zipfile, urllib.request
    url = "https://networks.skewed.de/net/karate/files/77.csv.zip"
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
    )
    # The "groups" property is in the last meaningful column
    src, trg = edges["source"].values, edges["target"].values
    n_nodes = len(nodes)
    adj = sparse.csr_matrix(
        (np.ones(len(src)), (src, trg)), shape=(n_nodes, n_nodes)
    )
    labels = np.unique(nodes.iloc[:, -1].values, return_inverse=True)[1]
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
