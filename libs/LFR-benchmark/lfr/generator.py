# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-01-21 17:11:06
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-15 12:50:28
import os
import pathlib
import random
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse


class NetworkGenerator:
    """
    A Python wrapper for LFR benchmark network generator.

    Attributes
    ----------
    None

    Parameters
    ----------
    N : int
        Number of nodes.
    k : int
        Average degree.
    maxk : int
        Maximum degree.
    minc : int
        Minimum community size.
    maxc : int
        Maximum community size.
    tau : float
        Degree exponent.
    tau2 : float
        Community size exponent.
    mu : float
        Mixing rate.

    Returns
    -------
    dict
        A dictionary containing the generated network with keys:
            'net': a sparse matrix representing the adjacency matrix of the generated network.
            'community_table': a pandas dataframe storing the community assignments of each node.
            'seed': a random seed used in the generation process.

    """

    def __init__(self):
        pass

    def generate_lfr_net(self, N, k, maxk, minc, maxc, tau, tau2, mu):
        with tempfile.TemporaryDirectory() as tmpdirname:
            root = pathlib.Path(__file__).parent.absolute()
            seed = random.randint(1, 1e3)
            with open(f"{tmpdirname}/time_seed.dat", "w", encoding="utf-8") as f:
                f.write("%d" % seed)
                f.close()

            # Run the command
            t1, t2 = tau, tau2
            os.system(
                f"cd {tmpdirname} && {root}/benchmark -N {N} -k {k} -maxk {maxk} -t1 {t1} -t2 {t2} -mu {mu} -minc {minc} -maxc {maxc} > /dev/null 2>&1"
            )

            edges = pd.read_csv(
                f"{tmpdirname}/network.dat", sep="\t", header=None
            ).values
            edges = edges - 1  # because the node id start from 1
            edges = pd.DataFrame(edges, columns=["source", "target"])

            communities = pd.read_csv(
                "{tmp}/community.dat".format(tmp=tmpdirname), sep="\t", header=None
            ).values
            communities[:, 0] -= 1  # because the node id start from 1
            communities = pd.DataFrame(communities, columns=["node_id", "community_id"])

        return edges, communities, seed

    def generate(
        self,
        N,  # number of nodes
        k,  # average degree
        maxk,  # maximum degree
        minc,  # minimum community size
        maxc,  # maximum community size
        tau,  # degree exponent
        tau2,  # community size exponent
        mu,  # Mixing rate
    ):
        """ "
        Generate a network using the Lancichinetti–Fortunato–Radicchi (LFR) model.

        Parameters
        ----------
        self : object
            The object instance.
        N : int
            Number of nodes in the generated network.
        k : float
            Average degree of each node.
        maxk : int
            Maximum degree of each node.
        minc : int
            Minimum size of communities.
        maxc : int
            Maximum size of communities.
        tau : float
            Power-law exponent for generating node degrees.
        tau2 : float
            Power-law exponent for generating community sizes.
        mu : float
            Mixing parameter that controls the fraction of intra-community edges to total edges.

        Returns
        -------
        dict
            A dictionary containing the generated network, community table, and seed for the random number generator used in the generation process.
        """
        edge_table, community_table, seed = self.generate_lfr_net(
            N=N, k=k, maxk=maxk, minc=minc, maxc=maxc, tau=tau, tau2=tau2, mu=mu
        )

        N = community_table.shape[0]
        A = sparse.csr_matrix(
            (
                np.ones(edge_table.shape[0]),
                (edge_table["source"], edge_table["target"]),
            ),
            shape=(N, N),
        )

        networks = {
            "net": A,
            "community_table": community_table,
            "seed": seed,
        }
        return networks
