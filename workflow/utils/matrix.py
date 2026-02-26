"""Shared matrix utilities for workflow scripts."""
import numpy as np
from scipy import sparse


def row_normalize(mat, mode="prob"):
    """Normalize a sparse CSR matrix row-wise.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
    mode : {"prob", "norm"}
        "prob" normalizes each row to sum to 1.
        "norm" normalizes each row to unit L2 norm.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    if mode == "prob":
        denom = np.array(mat.sum(axis=1)).reshape(-1).astype(float)
        return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat
    elif mode == "norm":
        denom = np.sqrt(np.array(mat.multiply(mat).sum(axis=1)).reshape(-1))
        return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat
    return np.nan
