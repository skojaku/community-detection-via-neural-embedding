# %%
import numpy as np
from scipy import sparse
import pandas as pd
import numpy as np
import sys
import graph_tool.all as gt


def get_largest_component(func):
    def wrapper(*args, **kwargs):
        net, node_labels = func(*args, **kwargs)
        n_components, labels = sparse.csgraph.connected_components(net, directed=False)
        largest_component_label = np.argmax(np.bincount(labels))
        largest_component_mask = np.where(labels == largest_component_label)[0]
        net = net[largest_component_mask, :][:, largest_component_mask]
        net = net + net.T
        net.data = np.ones(net.nnz)
        node_labels = node_labels[largest_component_mask]
        node_labels = np.unique(node_labels, return_inverse=True)[1]
        return net, node_labels

    return wrapper


# Airport network
@get_largest_component
def load_airport():
    node_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/adv-net-sci-course/main/data/airport_network_v2/node_table.csv"
    )
    edge_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/adv-net-sci-course/main/data/airport_network_v2/edge_table.csv"
    )
    src, trg = tuple(edge_table[["src", "trg"]].values.T)
    A = sparse.csr_matrix(
        (np.ones(len(src)), (src, trg)), shape=(len(node_table), len(node_table))
    )
    A = A + A.T
    A.data = np.ones(A.nnz)

    com_labels = node_table["region"].values
    return A, com_labels


@get_largest_component
def load_polblog():

    # Load the blog net
    edges = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/out.moreno_blogs_blogs"
    )
    node_table = pd.read_csv(
        "https://raw.githubusercontent.com/skojaku/core-periphery-detection/master/data/ent.moreno_blogs_blogs.blog.orientation",
        sep=",",
        header=None,
        names=["class"],
    )
    N = np.max(edges.max().values) + 1
    net = sparse.csc_matrix(
        (np.ones(edges.shape[0]), (edges.source - 1, edges.target - 1)), shape=(N, N)
    )
    return net, node_table["class"].values


@get_largest_component
def load_football():

    g = gt.collection.ns["football"]

    labels = np.array(g.vp["value"].get_array())
    net = sparse.csr_matrix(
        (np.ones(g.num_edges()), (g.get_edges().T)),
        shape=(g.num_vertices(), g.num_vertices()),
    )

    net = net + net.T
    net.data = np.ones(net.nnz)
    return net, labels


@get_largest_component
def load_highschool():

    g = gt.collection.ns["sp_high_school_new/2011"]

    labels = np.unique([g.vp["class"][i] for i in g.vertices()], return_inverse=True)[1]
    net = sparse.csr_matrix(
        (np.ones(g.num_edges()), (g.get_edges().T)),
        shape=(g.num_vertices(), g.num_vertices()),
    )

    net = net + net.T
    # net.data = np.ones(net.nnz)
    return net, labels


@get_largest_component
def load_polbooks():
    g = gt.collection.ns["polbooks"]

    labels = np.unique([g.vp["value"][i] for i in g.vertices()], return_inverse=True)[1]
    net = sparse.csr_matrix(
        (np.ones(g.num_edges()), (g.get_edges().T)),
        shape=(g.num_vertices(), g.num_vertices()),
    )

    net = net + net.T
    net.data = np.ones(net.nnz)
    return net, labels


@get_largest_component
def load_cora_citation_network():

    edge_table = pd.read_csv(
        "https://raw.githubusercontent.com/sk-classroom/GraphNeuralNetworks/master/derived/preprocess/cora.cites",
        sep="\t",
        header=None,
        names=["src", "trg"],
    )  # .to_csv("cora.cites", sep=' ', header=None, index=False)

    node_table = pd.read_csv(
        "https://raw.githubusercontent.com/sk-classroom/GraphNeuralNetworks/master/derived/preprocess/cora.content",
        sep="\t",
        header=None,
    )  # .to_csv("cora.cites", sep=' ', header=None, index=False)
    node_table = (
        node_table.iloc[:, np.array([0, -1])]
        .rename(columns={0: "node_id", 1434: "field"})
        .sort_values(by="node_id")
    )
    unode_ids, node_ids = np.unique(node_table["node_id"].values, return_inverse=True)
    toNewIndex = dict(zip(unode_ids, node_ids))
    edge_table["src"] = edge_table["src"].map(toNewIndex)
    edge_table["trg"] = edge_table["trg"].map(toNewIndex)

    node_table["node_id"] = node_ids

    net = sparse.csr_matrix(
        (np.ones(len(edge_table)), (edge_table["src"], edge_table["trg"])),
        shape=(len(node_table), len(node_table)),
    )
    net = net + net.T
    net.data = np.ones(net.nnz)
    labels = node_table["field"].values
    return net, labels


@get_largest_component
def load_karate():
    g = gt.collection.ns["karate/77"]

    labels = np.unique([g.vp["groups"][i] for i in g.vertices()], return_inverse=True)[
        1
    ]
    net = sparse.csr_matrix(
        (np.ones(g.num_edges()), (g.get_edges().T)),
        shape=(g.num_vertices(), g.num_vertices()),
    )
    return net, labels


network_generator = {
    "polblog": load_polblog,
    "cora": load_cora_citation_network,
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
    n = 10000
    K = 64
    cave = 10
    mu = 0.4
    output_file = ""
    params = {}
    params["netdata"] = "polblog"


data_name = params["netdata"]

net, labels = network_generator[data_name]()

num_components, _ = sparse.csgraph.connected_components(net, directed=False)
assert num_components == 1, "The network must have exactly one connected component."
assert net.shape[0] == len(labels)
assert (net != net.T).nnz == 0, "The network must be symmetric."


sparse.save_npz(output_file, net)
node_table = pd.DataFrame({"node_id": np.arange(len(labels)), "membership": labels})
node_table.to_csv(output_node_file, index=False)
