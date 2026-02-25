"""Shared I/O utilities for workflow scripts."""
import glob
import pathlib

import pandas as pd
from tqdm import tqdm


def load_files(dirname):
    """Load CSV result files from a directory or list of paths.

    Parses parameter key=value pairs encoded in filenames (separated by '~').

    Parameters
    ----------
    dirname : str or list
        Directory path (glob pattern) or explicit list of file paths.

    Returns
    -------
    pd.DataFrame
        Concatenated data with filename-encoded parameters merged in.
    """
    if isinstance(dirname, str):
        input_files = list(glob.glob(dirname + "/*"))
    else:
        input_files = list(dirname)

    def _get_params(filename, sep="~"):
        params = pathlib.Path(filename).stem.split("_")
        retval = {"filename": filename}
        for p in params:
            if sep not in p:
                continue
            kv = p.split(sep)
            retval[kv[0]] = kv[1]
        return retval

    df = pd.DataFrame([_get_params(f) for f in input_files])
    filenames = df["filename"].drop_duplicates().values
    dglist = []
    for filename in tqdm(filenames):
        dg = pd.read_csv(filename)
        dg["filename"] = filename
        dglist.append(dg)
    dg = pd.concat(dglist)
    return pd.merge(df, dg, on="filename")


def to_numeric(df, to_int, to_float):
    """Convert DataFrame columns to int or float types.

    Parameters
    ----------
    df : pd.DataFrame
    to_int : list of str
        Column names to cast to int (first cast to float, then int).
    to_float : list of str
        Column names to cast to float.

    Returns
    -------
    pd.DataFrame
    """
    existing_float = [k for k in to_int + to_float if k in df.columns]
    existing_int = [k for k in to_int if k in df.columns]
    df = df.astype({k: float for k in existing_float}, errors="ignore")
    df = df.astype({k: int for k in existing_int}, errors="ignore")
    return df
