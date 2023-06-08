import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse

sys.path.append(os.path.abspath(os.path.join("./libs/lfr_benchmark")))
from lfr_benchmark.generator import NetworkGenerator as NetworkGenerator

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
}


root = Path().parent.absolute()
ng = NetworkGenerator()
data = ng.generate(params, mu)
os.chdir(root)

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
pd.DataFrame({"node_id": np.arange(len(community_ids)), "membership": community_ids}).to_csv(
    output_node_file, index=False
)
