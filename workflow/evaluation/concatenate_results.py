# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-13 16:13:54
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-02-02 03:50:54
"""Putting together results into a data table."""
# %%
import glob
import pathlib
import sys

import pandas as pd
from tqdm import tqdm


def load_files(dirname):
    if isinstance(dirname, str):
        input_files = list(glob.glob(dirname))
    else:
        input_files = dirname

    def get_params(filenames):
        def _get_params(filename, sep="~"):
            params = pathlib.Path(filename).stem.split("_")
            retval = {"filename": filename}
            for p in params:
                if sep not in p:
                    continue
                kv = p.split(sep)

                retval[kv[0]] = kv[1]
            return retval

        return pd.DataFrame([_get_params(filename) for filename in filenames])

    df = get_params(input_files)
    filenames = df["filename"].drop_duplicates().values
    dglist = []
    for filename in tqdm(filenames):
        dg = pd.read_csv(filename)
        dg["filename"] = filename
        dglist += [dg]
    dg = pd.concat(dglist)
    df = pd.merge(df, dg, on="filename")
    return df


def to_numeric(df, to_int, to_float):
    df = df.astype({k: float for k in to_int + to_float}, errors="ignore")
    df = df.astype({k: int for k in to_int}, errors="ignore")
    return df


if "snakemake" in sys.modules:
    input_files = snakemake.input["input_files"]
    output_file = snakemake.output["output_file"]
    to_int = snakemake.params["to_int"]
    to_float = snakemake.params["to_float"]
else:
    # mlt
    #input_files = "../../data/multi_partition_model/evaluations/score*.npz"
    #output_file = "../../data/multi_partition_model/all-result.csv"
    #to_int = ["n", "K", "dim", "sample", "length", "dim", "cave"]
    #to_float = ["mu"]
    # lfr
    input_files = "../../data/lfr/evaluations/score*.npz"
    output_file = "../../data/lfr/all-result.csv"
    to_int=["n", "k", "tau2", "minc", "dim", "sample", "length", "dim"]
    to_float=["mu", "tau"]

#%% Load
data_table = load_files(input_files).fillna("")

# %% Type conversion
data_table = to_numeric(data_table, to_int, to_float)
data_table = data_table.rename(columns={"K": "q"})

# %% Save
data_table.to_csv(output_file)
