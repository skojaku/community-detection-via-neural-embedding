# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-02 13:34:28
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-23 05:31:38
import os
import pathlib
import tempfile
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx


def detect(
    A, q, iters=1, dumping_rate=1.0, init_memberships=None, params_sbm="", mute=True
):
    """
    Perform belief propagation.

    Parameters
    ----------
    A : array_like or sparse matrix
        The adjacency matrix of the graph to be clustered.
    q : float
        "Temperature" parameter for the clustering algorithm.
    iters : int, optional (default: 1)
        Number of iterations to run the clustering algorithm.
    dumping_rate : float, optional (default: 1.0)
        Damping rate for the belief propagation algorithm.
    init_memberships : array_like, optional (default: None)
        Initial cluster assignments for the nodes in the graph. If specified,
        this must be a one-dimensional array-like object with length equal to
        the number of nodes in the graph.
    params_sbm : str, optional (default: "")
        Additional command line arguments to pass to the SBM executable.
    mute : bool, optional (default: True)
        Whether to suppress standard output from the SBM executable.

    Returns
    -------
    cids : ndarray
        One-dimensional array of integers representing the cluster assignments
        for the nodes in the input graph. The length of this array is equal to
        the number of nodes in the graph.
    """
    if init_memberships is not None:
        n_nodes = A.shape[0]
        l, memberships = np.unique(init_memberships, return_inverse=True)
        K = len(l)
        U = sparse.csr_matrix(
            (np.ones_like(memberships), (np.arange(n_nodes), memberships)),
            shape=(n_nodes, K),
        )
        Nc = np.array(U.sum(axis=0)).reshape(-1)
        cab_init = np.maximum(n_nodes * ((U.T @ A @ U).toarray() / np.outer(Nc, Nc)), 1)
        p_init = Nc / n_nodes
    else:
        cab_init, p_init = None, None

    A.setdiag(0)
    A.eliminate_zeros()
    A.data = A.data * 0 + 1
    A = sparse.csr_matrix(A)
    G = nx.from_scipy_sparse_array(A)

    for n1, n2, d in G.edges(data=True):
        d.clear()
    for n1, d in G.nodes(data=True):
        d.clear()
    with tempfile.TemporaryDirectory() as tmpdirname:
        root = pathlib.Path(__file__).parent.absolute()
        # graph_file_name = f"./graph.gml"
        # cab_file_name = f"./tmp.cab"
        # output_file_name = f"./tmp.csv"
        graph_file_name = f"{tmpdirname}/graph.gml"
        cab_file_name = f"{tmpdirname}/tmp.cab"
        output_file_name = f"{tmpdirname}/tmp.csv"

        nx.write_gml(G, graph_file_name)
        if cab_init is not None:
            generate_cab_file(cab_file_name, p_init, cab_init)
            params_sbm += f" -L {cab_file_name}"

        if mute:
            mute = ">/dev/null"
        else:
            mute = ""

        energy = np.inf
        cids = np.zeros(A.shape[0], dtype=int)
        for _ in range(iters):
            os.system(
                f"{root}/sbm learn -q {q} -R {dumping_rate} -l {graph_file_name} -w {output_file_name} {params_sbm} {mute}"
            )
            df = pd.read_csv(output_file_name)
            energy_t = df["energy"].values[0]
            if energy > energy_t:
                energy = energy_t
                cids = df["block"].values
                cids = np.unique(cids, return_inverse=True)[1]

    return cids


def generate_cab_file(filename, p, cab):
    pstr = " ".join([f"{d:.4f}" for d in p])
    cabstr = "\n".join(
        [
            " ".join([f"{cab[r,c]:.4f}" for c in range(cab.shape[1])])
            for r in range(cab.shape[0])
        ]
    )
    s = f"#Vector_na:\n{pstr}\n#Matrix_cab:\n{cabstr}"
    text_file = open(filename, "w")
    text_file.write(s)
    text_file.close()
    return
