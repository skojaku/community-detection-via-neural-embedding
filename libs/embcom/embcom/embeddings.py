# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-08-26 09:51:23
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-15 12:02:03
"""Module for embedding."""
# %%
import gensim
import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from embcom import samplers, utils


# Base class
#
class NodeEmbeddings:
    """Super class for node embedding class."""

    def __init__(self):
        self.in_vec = None
        self.out_vec = None

    def fit(self):
        """Estimating the parameters for embedding."""
        pass

    def transform(self, dim, return_out_vector=False):
        """Compute the coordinates of nodes in the embedding space of the
        prescribed dimensions."""
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that for the previous call of transform function
        if self.out_vec is None:
            self.update_embedding(dim)
        elif self.out_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.out_vec if return_out_vector else self.in_vec

    def update_embedding(self, dim):
        """Update embedding."""
        pass


class Node2Vec(NodeEmbeddings):
    """A python class for the node2vec.

    Parameters
    ----------
    num_walks : int (optional, default 10)
        Number of walks per node
    walk_length : int (optional, default 40)
        Length of walks
    window_length : int (optional, default 10)
        Restart probability of a random walker.
    p : node2vec parameter (TODO: Write doc)
    q : node2vec parameter (TODO: Write doc)
    """

    def __init__(
        self,
        num_walks=10,
        walk_length=80,
        window_length=10,
        p=1.0,
        q=1.0,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length
        self.sampler = samplers.Node2VecWalkSampler(
            num_walks=num_walks,
            walk_length=walk_length,
            p=p,
            q=q,
        )

        self.sentences = None
        self.model = None
        self.verbose = verbose

        self.w2vparams = {
            "sg": 1,
            "min_count": 0,
            "epochs": 1,
            "workers": 4,
        }

    def fit(self, net):
        """Estimating the parameters for embedding.

        Parameters
        ---------
        net : nx.Graph object
            Network to be embedded. The graph type can be anything if
            the graph type is supported for the node samplers.

        Return
        ------
        self : Node2Vec
        """
        A = utils.to_adjacency_matrix(net)
        self.sampler.sampling(A)
        return self

    def update_embedding(self, dim):
        # Update the dimension and train the model
        # Sample the sequence of nodes using a random walk
        self.w2vparams["window"] = self.window_length

        self.w2vparams["vector_size"] = dim
        self.model = gensim.models.Word2Vec(
            sentences=self.sampler.walks, **self.w2vparams
        )

        num_nodes = len(self.model.wv.index_to_key)
        self.in_vec = np.zeros((num_nodes, dim))
        self.out_vec = np.zeros((num_nodes, dim))
        for i in range(num_nodes):
            if i not in self.model.wv:
                continue
            self.in_vec[i, :] = self.model.wv[i]
            self.out_vec[i, :] = self.model.syn1neg[self.model.wv.key_to_index[i]]


class DeepWalk(Node2Vec):
    def __init__(self, **params):
        Node2Vec.__init__(self, **params)
        self.w2vparams = {
            "sg": 0,
            "hs": 1,
            "min_count": 0,
            "workers": 4,
        }


class LaplacianEigenMap(NodeEmbeddings):
    def __init__(self, p=100, q=40, reconstruction_vector=False):
        self.in_vec = None
        self.L = None
        self.deg = None
        self.p = p
        self.q = q
        self.reconstruction_vector = reconstruction_vector

    def fit(self, G):
        A = utils.to_adjacency_matrix(G)

        # Compute the (inverse) normalized laplacian matrix
        deg = np.array(A.sum(axis=1)).reshape(-1)
        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
        L = Dsqrt @ A @ Dsqrt

        self.L = L
        self.deg = deg
        return self

    def transform(self, dim, return_out_vector=False):
        if self.in_vec is None:
            self.update_embedding(dim)
        elif self.in_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.in_vec

    def update_embedding(self, dim):
        if self.reconstruction_vector:
            s, u = sparse.linalg.eigs(self.L, k=dim, which="LR")
            s, u = np.real(s), np.real(u)
            order = np.argsort(s)[::-1]
            self.in_vec = u[:, order]
            self.out_vec = u[:, order]
        else:
            s, u = sparse.linalg.eigs(self.L, k=dim + 1, which="LR")
            s, u = np.real(s), np.real(u)
            order = np.argsort(-s)[1:]
            s, u = s[order], u[:, order]
            Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(self.deg), 1e-12), format="csr")
            self.in_vec = Dsqrt @ u @ sparse.diags(np.sqrt(np.abs(s)))
            self.out_vec = u


class AdjacencySpectralEmbedding(NodeEmbeddings):
    def __init__(
        self,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        return self

    def update_embedding(self, dim):
        svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
        u = svd.fit_transform(self.A)
        s = svd.singular_values_
        self.in_vec = u @ sparse.diags(s)


class ModularitySpectralEmbedding(NodeEmbeddings):
    def __init__(self, verbose=False, reconstruction_vector=False, p=100, q=40):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.reconstruction_vector = reconstruction_vector
        self.p = p
        self.q = q

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):
        s, u = sparse.linalg.eigs(self.A, k=dim + 1, which="LR")
        s, u = np.real(s), np.real(u)
        s = s[1:]
        u = u[:, 1:]

        if self.reconstruction_vector:
            is_positive = s > 0
            u[:, ~is_positive] = 0
            s[~is_positive] = 0
            self.in_vec = u @ sparse.diags(np.sqrt(s))
        else:
            self.in_vec = u @ sparse.diags(np.sqrt(np.abs(s)))
        self.out_vec = u


class LinearizedNode2Vec(NodeEmbeddings):
    def __init__(self, verbose=False, window_length=10, p=100, q=40):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.window_length = window_length
        self.p = p
        self.q = q

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):
        # Calculate the normalized transition matrix
        Dinvsqrt = sparse.diags(1 / np.sqrt(np.maximum(1, self.deg)))
        Psym = Dinvsqrt @ self.A @ Dinvsqrt

        s, u = sparse.linalg.eigs(Psym, k=dim + 1, which="LR")
        s, u = np.real(s), np.real(u)
        order = np.argsort(-s)
        s, u = s[order], u[:, order]

        s = np.abs(s)
        mask = s < np.max(s)
        u = u[:, mask]
        s = s[mask]

        if self.window_length > 1:
            s = (s * (1 - s**self.window_length)) / (self.window_length * (1 - s))

        self.in_vec = u @ sparse.diags(np.sqrt(np.abs(s)))
        self.out_vec = None


class NonBacktrackingSpectralEmbedding(NodeEmbeddings):
    def __init__(self, verbose=False, auto_dim=False):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.auto_dim = auto_dim
        self.C = 10

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        self.A = A
        return self

    def update_embedding(self, dim):
        N = self.A.shape[0]
        Z = sparse.csr_matrix((N, N))
        I = sparse.identity(N, format="csr")
        D = sparse.diags(self.deg)
        B = sparse.bmat([[Z, D - I], [-I, self.A]], format="csr")

        if self.auto_dim is False:
            s, v = sparse.linalg.eigs(B, k=dim, tol=1e-4)
            s, v = np.real(s), np.real(v)
            order = np.argsort(-np.abs(s))
            s, v = s[order], v[:, order]
            v = v[N:, :]

            # Normalize the eigenvectors because we cut half the vec
            # and omit the imaginary part.
            c = np.array(np.linalg.norm(v, axis=0)).reshape(-1)
            v = v @ np.diag(1 / c)

            # Weight the dimension by the eigenvalues
            v = v @ np.diag(np.sqrt(np.abs(s)))
        else:
            dim = int(self.C * np.sqrt(N))
            dim = np.minimum(dim, N - 1)

            s, v = sparse.linalg.eigs(B, k=dim + 1, tol=1e-4)

            c = int(self.A.sum() / N)
            s, v = s[np.abs(s) > c], v[:, np.abs(s) > c]

            order = np.argsort(s)
            s, v = s[order], v[:, order]
            s, v = s[1:], v[:, 1:]
            v = v[N:, :]
            c = np.array(np.linalg.norm(v, axis=0)).reshape(-1)
            v = v @ np.diag(1 / c)

        self.in_vec = v
