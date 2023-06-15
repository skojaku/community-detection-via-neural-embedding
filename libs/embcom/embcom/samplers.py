"""This module contains sampler class which generates a sequence of nodes from
a network using a random walk.

All samplers should be the subclass of NodeSampler class.
"""
from abc import ABCMeta, abstractmethod
from numba import njit
import numpy as np

class NodeSampler(metaclass=ABCMeta):
    """Super class for node sampler class.

    Implement
        - sampling

    Optional
        - get_decomposed_trans_matrix
    """

    @abstractmethod
    def sampling(self):
        """Generate a sequence of walks over the network.

        Return
        ------
        walks : numpy.ndarray (number_of_walks, number_of_steps)
            Each row indicates a trajectory of a random walker
            walk[i,j] indicates the jth step for walker i.
        """

class Node2VecWalkSampler(NodeSampler):
    def __init__(
        self,
        num_walks=10,
        walk_length=80,
        p=1.0,
        q=1.0,
        **params
    ):
        """Noe2VecWalk Sampler

        Parameters
        ----------
        num_walks : int (Optional; Default 1)
            Number of walkers to simulate for each randomized network.
            A larger value removes the bias better but takes more time.
        walk_length : int
            Number of steps for a single walker
        p : float
            Parameter for the node2vec
        q : float
            Parameter for the node2vec
        window_length : int
            Size of the window
        verbose : bool
            Set true to display the progress (NOT IMPLEMENTED)
        """
        self.num_nodes = -1

        # parameters for random walker
        self.num_walks = int(num_walks)
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.walks = None

    def sampling(self, net):
        self.num_nodes = net.shape[0]
        self.A = net

        self.walks = simulate_node2vec_walk(
            A = self.A,
            num_walks = self.num_walks,
            walk_length = self.walk_length,
            start_node_ids = None,
            p = self.p,
            q = self.q,
        )

#
# SimpleWalk Sampler
#
class SimpleWalkSampler(Node2VecWalkSampler):
    def __init__(
        self,
        **params
    ):
        """Simple walk without bias

        Parameters
        ----------
        num_walks : int (Optional; Default 1)
            Number of walkers to simulate for each randomized network.
            A larger value removes the bias better but takes more time.
        walk_length : int
            Number of steps for a single walker
        window_length : int
            Size of the window
        verbose : bool
            Set true to display the progress (NOT IMPLEMENTED)
        """
        params["p"] = 1
        params["q"] = 1
        Node2VecWalkSampler.__init__(self, **params)


#
# Non-backtracking walks
#
class NonBacktrackingWalkSampler(NodeSampler):
    def __init__(
        self, num_walks=10, walk_length=80, **params
    ):
        self.num_nodes = -1
        self.num_walks = int(num_walks)
        self.walk_length = walk_length
        self.walks = None
        self.W = None

    def sampling(self, net):
        self.num_nodes = net.shape[0]
        self.A = net
        self.walks = []
        target_walk_num = self.walk_length * self.num_walks * self.num_nodes
        total_walk_num = 0
        while total_walk_num < target_walk_num:
            _walks = simulate_non_backtracking_walk(
                self.A,
                self.num_walks,
                self.walk_length,
            )
            for _walk in _walks:
                self.walks.append(_walk)
                total_walk_num+=len(_walk)
                if total_walk_num > target_walk_num:
                    break



#
# node2vec walk
#
def simulate_node2vec_walk(
    A,
    num_walks,
    walk_length,
    start_node_ids = None,
    p = 1,
    q = 1,
):
    if A.getformat() != "csr":
        raise TypeError("A should be in the scipy.sparse.csc_matrix")

    if start_node_ids is None:
        start_node_ids = np.arange(A.shape[0])

    is_weighted = np.max(A.data) != np.min(A.data)
    A.sort_indices()
    if is_weighted:
        data = A.data / A.sum(axis=1).A1.repeat(np.diff(A.indptr))
        A.data = _csr_row_cumsum(A.indptr, data)

    walks = []
    for _ in range(num_walks):
        for start in start_node_ids:
            if is_weighted:
                walk = _random_walk_weighted(
                    A.indptr, A.indices, A.data, walk_length, p, q, start
                )
            else:
                walk = _random_walk(A.indptr, A.indices, walk_length, p, q, start)
            walks.append(walk.tolist())

    return walks

@njit(nogil=True)
def _csr_row_cumsum(indptr, data):
    out = np.empty_like(data)
    for i in range(len(indptr) - 1):
        acc = 0
        for j in range(indptr[i], indptr[i + 1]):
            acc += data[j]
            out[j] = acc
        out[j] = 1.0
    return out

@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _isin_sorted(a, x):
    return a[np.searchsorted(a, x)] == x


@njit(nogil=True)
def _random_walk(indptr, indices, walk_length, p, q, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    neighbors = _neighbors(indptr, indices, t)
    if not neighbors.size:
        return walk[:1]
    walk[1] = np.random.choice(_neighbors(indptr, indices, t))
    for j in range(2, walk_length):
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        if not neighbors.size:
            return walk[:j]
        if p == q == 1:
            # faster version
            walk[j] = np.random.choice(neighbors)
            continue
        while True:
            new_node = np.random.choice(neighbors)
            r = np.random.rand()
            if new_node == walk[j - 2]:
                if r < prob_0:
                    break
            elif _isin_sorted(_neighbors(indptr, indices, walk[j - 2]), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk[j] = new_node
    return walk


@njit(nogil=True)
def _random_walk_weighted(indptr, indices, data, walk_length, p, q, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    neighbors = _neighbors(indptr, indices, t)
    if not neighbors.size:
        return walk[:1]
    walk[1] = _neighbors(indptr, indices, t)[
        np.searchsorted(_neighbors(indptr, data, t), np.random.rand())
    ]
    for j in range(2, walk_length):
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        if not neighbors.size:
            return walk[:j]
        neighbors_p = _neighbors(indptr, data, walk[j - 1])
        if p == q == 1:
            # faster version
            walk[j] = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
            continue
        while True:
            new_node = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
            r = np.random.rand()
            if new_node == walk[j - 2]:
                if r < prob_0:
                    break
            elif _isin_sorted(_neighbors(indptr, indices, walk[j - 2]), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk[j] = new_node
    return walk


#
# non-backtracking walk
#
def simulate_non_backtracking_walk(
    A,
    num_walks,
    walk_length,
    start_node_ids = None,
):
    """Wrapper for."""

    if A.getformat() != "csr":
        raise TypeError("A should be in the scipy.sparse.csc_matrix")

    if start_node_ids is None:
        start_node_ids = np.arange(A.shape[0])

    is_weighted = np.max(A.data) != np.min(A.data)
    A.sort_indices()
    if is_weighted:
        data = A.data / A.sum(axis=1).A1.repeat(np.diff(A.indptr))
        A.data = _csr_row_cumsum(A.indptr, data)

    walks = []
    for _ in range(num_walks):
        for start in start_node_ids:
            if is_weighted:
                walk = _non_backtracking_random_walk_weighted(
                    A.indptr, A.indices, A.data, walk_length, start
                )
            else:
                walk = _non_backtracking_random_walk(A.indptr, A.indices, walk_length, start)
            walks.append(walk.tolist())

    return walks

@njit(nogil=True)
def _non_backtracking_random_walk(indptr, indices, walk_length, t):
    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    neighbors = _neighbors(indptr, indices, t)
    if not neighbors.size:
        return walk[:1]
    walk[1] = np.random.choice(_neighbors(indptr, indices, t))
    for j in range(2, walk_length):
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        if not neighbors.size:
            return walk[:j]
        elif (neighbors.size == 1) & (neighbors[0] == walk[j - 2]):
            return walk[:j]
        else:
            # find a neighbor by a roulette selection
            new_node = walk[j - 2]
            while new_node == walk[j - 2]:
                new_node = np.random.choice(neighbors)
            walk[j] = new_node
    return walk


@njit(nogil=True)
def _non_backtracking_random_walk_weighted(indptr, indices, data, walk_length, t):
    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    neighbors = _neighbors(indptr, indices, t)
    if not neighbors.size:
        return walk[:1]
    walk[1] = _neighbors(indptr, indices, t)[
        np.searchsorted(_neighbors(indptr, data, t), np.random.rand())
    ]
    for j in range(2, walk_length):
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        if not neighbors.size:
            return walk[:j]
        elif (neighbors.size == 1) & (neighbors[0] == walk[j - 2]):
            return walk[:j]
        else:
            neighbors_p = _neighbors(indptr, data, walk[j - 1])
            new_node = walk[j - 2]
            while new_node == walk[j - 2]:
                new_node = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
            walk[j] = new_node
    return walk