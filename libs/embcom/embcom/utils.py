# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-08 16:38:19
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-15 11:56:36
import networkx as nx
import numpy as np
from scipy import sparse


def to_trans_mat(mat):
    """
    Computes the transition matrix of a given Markov chain represented as a square matrix.

    Parameters
    ----------
    mat : numpy.ndarray
        The input matrix, which represents the transition probabilities of the Markov chain.

    Returns
    -------
    scipy.sparse.csr_matrix
        The sparse CSR matrix representing the transition matrix of the input Markov chain.
    """
    denom = np.array(mat.sum(axis=1)).reshape(-1).astype(float)
    return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat


def pairing(k1, k2, unordered=False):
    """
    Cantor pairing function.

    Parameters
    ----------
    k1 : int
        First integer to pair.
    k2 : int
        Second integer to pair.
    unordered : bool, optional (default=False)
        Whether to compute the pairing for unordered pairs (i.e., (k1,k2) and (k2,k1) are considered equivalent).

    Returns
    -------
    int
        The result of applying the Cantor pairing function on the input integers.
    """
    k12 = k1 + k2
    if unordered:
        return (k12 * (k12 + 1)) * 0.5 + np.minimum(k1, k2)
    else:
        return (k12 * (k12 + 1)) * 0.5 + k2


def depairing(z):
    """
    Inverse of Cantor pairing function.

    Parameters
    ----------
    z : int
        The output of the Cantor pairing function.

    Returns
    -------
    Tuple[int, int]
        A tuple of two integers, representing the original paired values used to produce the input z.
    """
    w = np.floor((np.sqrt(8 * z + 1) - 1) * 0.5)
    t = (w**2 + w) * 0.5
    y = np.round(z - t).astype(np.int64)
    x = np.round(w - y).astype(np.int64)
    return x, y


def to_adjacency_matrix(net):
    """
    Converts an input graph representation to its corresponding adjacency matrix.

    Parameters
    ----------
    net : {scipy.sparse.csr_matrix, networkx.classes.graph.Graph, numpy.ndarray}
        The input graph representation. If it's a numpy.ndarray, it will be converted to a sparse CSR matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        The sparse CSR matrix representing the adjacency matrix of the input graph.
    """
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net.asfptype()
        return sparse.csr_matrix(net).asfptype()
    elif "networkx" in "%s" % type(net):
        return nx.adjacency_matrix(net).asfptype()
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net).asfptype()


def safe_log(A, minval=1e-12):
    """
    Computes the element-wise logarithm of an input array, with a minimum value limit.

    Parameters
    ----------
    A : {numpy.ndarray, scipy.sparse.csr_matrix}
        The input array or sparse matrix to take the logarithm of.
    minval : float, optional (default=1e-12)
        The minimum allowed value for elements in the input array. Values below this threshold are set to this value.

    Returns
    -------
    {numpy.ndarray, scipy.sparse.csr_matrix}
        A new array/matrix containing the element-wise logarithm of the input with values clipped at `minval`.
    """
    if sparse.issparse(A):
        A.data = np.log(np.maximum(A.data, minval))
        return A
    else:
        return np.log(np.maximum(A, minval))


def matrix_sum_power(A, T):
    """
    Computes the sum of powers of an input matrix, up to a given exponent.

    Parameters
    ----------
    A : numpy.ndarray
        The input square matrix to exponentiate and sum.
    T : int
        The maximum exponent to compute the sum up to.

    Returns
    -------
    numpy.ndarray
        The resulting matrix obtained by computing the sum of powers of the input matrix A, up to exponent T.
    """
    At = np.eye(A.shape[0])
    As = np.zeros((A.shape[0], A.shape[0]))
    for _ in range(T):
        At = A @ At
        As += At
    return As
