import graph_tool.all as gt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def get_membership(n, q):
    return np.sort(np.arange(n) % q)


def generate_network(Cave, mixing_rate, N, q, memberships=None, **params):

    if memberships is None:
        memberships = get_membership(N, q)

    q = int(np.max(memberships) + 1)
    N = len(memberships)
    U = sparse.csr_matrix((np.ones(N), (np.arange(N), memberships)), shape=(N, q))
    Cin, Cout = get_cin_cout(Cave, mixing_rate, q)
    pout = Cout / N
    pin = Cin / N

    Nk = np.array(U.sum(axis=0)).reshape(-1)

    P = np.ones((q, q)) * pout + np.eye(q) * (pin - pout)
    probs = np.diag(Nk) @ P @ np.diag(Nk)

    gt_params = {
        "b": memberships,
        "probs": probs,
    }

    # Generate the network until the degree sequence
    # satisfied the thresholds
    while True:
        g = gt.generate_sbm(**gt_params)
        A = gt.adjacency(g).T
        A.data = np.ones_like(A.data)
        # check if the graph is connected
        if connected_components(A)[0] == 1:
            break
    return A, memberships


def make_node2vec_matrix(A, L, returnPt=False):
    n = A.shape[0]
    deg = np.array(A.sum(axis=0)).reshape(-1)
    P = sparse.diags(1 / np.maximum(deg, 1)) @ A
    pi = deg / np.sum(deg)
    logpi = np.log(pi)
    Prwr = calc_rwr(P.toarray(), L, returnPt=returnPt)
    R = np.log(Prwr) - np.outer(np.ones(n), logpi)
    R[np.isnan(R)] = 0
    R[np.isinf(R)] = 0
    return R


def get_cin_cout(Cave, mixing_rate, q, **params):
    cout = Cave * mixing_rate
    cin = q * Cave - (q - 1) * cout
    return cin, cout


def make_node2vec_matrix_limit(A, L, returnPt=False):
    deg = np.array(A.sum(axis=0)).reshape(-1)
    P = sparse.diags(1 / np.maximum(deg, 1)) @ A
    Prwr = calc_rwr(P.toarray(), L, returnPt=returnPt)
    R = np.sum(deg) * Prwr @ sparse.diags(1 / np.maximum(deg, 1)) - 1

    return R


def calc_rwr(P, L, returnPt=False):

    Pt = P.copy()
    Ps = None
    for t in range(L):
        if Ps is None:
            Ps = Pt / L
            continue
        Pt = P @ Pt
        Ps += Pt / L
    if returnPt:
        return Pt
    return Ps


def graph_kernel(X, phi):
    s, v = np.linalg.eig(X)
    s = phi(s)
    return v @ np.diag(s) @ v.T
