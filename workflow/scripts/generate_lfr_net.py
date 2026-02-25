"""Generate a synthetic network using the LFR benchmark model."""
import sys

import numpy as np
import pandas as pd
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
else:
    input_file = "../data/"
    output_file = "../data/"

# Default max degree and max community size scale as sqrt(10 * N)
maxk = int(np.sqrt(10 * N))
maxc = int(np.ceil(np.sqrt(N * 10)))

lfr_params = {
    "N": N,
    "k": k,
    "maxk": maxk,
    "minc": minc,
    "maxc": maxc,
    "tau": tau,
    "tau2": tau2,
    "mu": mu,
}

generator = lfr.NetworkGenerator()
result = generator.generate(**lfr_params)

net = result["net"]
community_table = result["community_table"]

# Convert 1-indexed community IDs to 0-indexed
community_ids = (
    community_table.sort_values(by="node_id")["community_id"].values.astype(int) - 1
)

# Save network and community membership
sparse.save_npz(output_net_file, net)
pd.DataFrame(
    {"node_id": np.arange(len(community_ids)), "membership": community_ids}
).to_csv(output_node_file, index=False)
