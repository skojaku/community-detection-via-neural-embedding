# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-08 16:37:34
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-15 13:13:15
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse
import lfr

if "snakemake" in sys.modules:
    params = snakemake.params["parameters"]
    N = float(params["n"])
    k = float(params["k"])
    tau = float(params["tau"])
    tau2 = float(params["tau2"])
    mu = float(params["mu"])
    minc = float(params["minc"])
    output_net_file = snakemake.output["output_file"]
    output_node_file = snakemake.output["output_node_file"]

    maxk = None
    maxc = None

    if (maxk is None) or (maxk == "None"):
        maxk = int(np.sqrt(10 * N))
    else:
        maxk = int(maxk)

    if (maxc is None) or (maxc == "None"):
        maxc = int(np.ceil(np.sqrt(N * 10)))
    else:
        maxc = int(maxc)

else:
    input_file = "../data/"
    output_file = "../data/"

params = {
    "N": N,
    "k": k,
    "maxk": maxk,
    "minc": minc,
    "maxc": maxc,
    "tau": tau,
    "tau2": tau2,
    "mu": mu,
}


ng = lfr.NetworkGenerator()
data = ng.generate(params)

# Load the network
net = data["net"]
community_table = data["community_table"]
params = data["params"]
seed = data["seed"]

community_ids = community_table.sort_values(by="node_id")["community_id"].values.astype(
    int
)
community_ids -= 1  # because the offset is one

# Save
sparse.save_npz(output_net_file, net)
pd.DataFrame(
    {"node_id": np.arange(len(community_ids)), "membership": community_ids}
).to_csv(output_node_file, index=False)
