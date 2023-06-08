"""Calculate the spectral density of the adjacency matrix of a bipartition
model."""
import sys

import numpy as np
import pandas as pd
import utils
from scipy import sparse
from tqdm import tqdm

if "snakemake" in sys.modules:
    param_net = snakemake.params["parameters"]
    L = int(param_net["L"])
    matrixType = param_net["matrixType"]
    n_samples = int(param_net["n_samples"])
    output_file = snakemake.output["output_file"]
else:
    output_file = "../data/"
    param_net = {"Cave": 50, "mixing_rate": 1 - 1 / np.sqrt(50), "N": 1000, "q": 2}
    L = 10  # window size
    n_samples = 10  # Number of samples

param_net["memberships"] = utils.get_membership(param_net["N"], param_net["q"])
return_xt = returnPt = False

#
# Graph kernel
#
def power_sum(x, L, return_xt=False):
    xs = 0
    xt = 1
    for i in range(L):
        xt *= x
        xs += xt
    xs /= L
    if return_xt:
        return xt
    return xs


def partial_power_sum(x, L, return_xt=False):
    xs = 0
    for t in range(1, L * L + 1):
        xs += ((-1) ** (t - 1)) * (x ** (t - 1))
    return xs / L


kernel_func = lambda x, l=L: power_sum(x, l, return_xt)

# Get the parameter of the bipartition model
cin, cout = utils.get_cin_cout(**param_net)


#%% Calculate the ensemble average and variances using simulations
net_list = [utils.generate_network(**param_net)[0] for i in tqdm(range(n_samples))]
R_list = [
    utils.make_node2vec_matrix_limit(net, L, returnPt=returnPt)
    if matrixType == "linearized-node2vec"
    else utils.make_node2vec_matrix(net, L, returnPt=returnPt)
    for net in tqdm(net_list)
]
Rave = np.mean([R for R in R_list], axis=0)
Rvar = np.var([R for R in R_list], axis=0)

# %% Simulate the eigenvalue distributions
u = ((2 * param_net["memberships"] - 1) / np.sqrt(param_net["N"])).reshape((-1, 1))
eigval_list, largest_eigval_list = [], []
for R in tqdm(R_list):
    # Decompose the matrix into the community and random components
    vals = np.linalg.eigvals(R - Rave)
    z1val, _ = sparse.linalg.eigs(R, k=1, which="LR")

    # Save
    z1val = np.real(z1val)
    eigval_list.append(np.real(vals))
    largest_eigval_list.append(z1val)

# Format the results as pandas DataFrame
simulated_data = pd.concat(
    [
        pd.DataFrame(
            {
                "eigs": np.concatenate(eigval_list),
                "prob_mass": 1,
                "dataType": "simulation",
                "component": "random",
            }
        ),
        pd.DataFrame(
            {
                "eigs": np.concatenate(largest_eigval_list),
                "prob_mass": 1,
                "dataType": "simulation",
                "component": "community",
            }
        ),
    ],
)

# %% Calculate the support and density of the eigenvalue distribution analytically
# Intermediate variable
d = (cin + cout) / 2
n = param_net["N"]
nbins = 100

# Calculate the support
sigY = 1 / (n * d)
support = (
    param_net["N"] * kernel_func(-2 * np.sqrt(n * sigY)),
    param_net["N"] * kernel_func(2 * np.sqrt(n * sigY)),
)

# Calculate the semi-circle density function
ylim = 2 * np.sqrt(n * sigY)
ys = np.linspace(-ylim, ylim, nbins)
density = (
    n
    * np.sqrt(4 * n * sigY - ys ** 2)
    / (2 * np.pi * n * sigY)
    * partial_power_sum(ys, L)
)
density[np.isnan(density)] = 0

# Transform to the space of z from y
zs = param_net["N"] * kernel_func(ys)

analytical_results = pd.DataFrame(
    {"eigs": zs, "prob_mass": density, "dataType": "analytical", "component": "random"}
)

# %% Save
pd.concat([simulated_data, analytical_results]).to_csv(output_file, index=False)
