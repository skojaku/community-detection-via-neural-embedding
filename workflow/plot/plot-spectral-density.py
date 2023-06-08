# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-17 06:56:49
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-17 06:56:55
# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
else:
    input_file = "../../data/multi_partition_model/spectral_analysis/Cave~50_mixing_rate~0.5_N~1000_q~2_matrixType~node2vec_L~50_n_samples~10.csv"
    output_file = "../data/"

# %%
# Load data
#
data_table = pd.read_csv(input_file)

# Classify data
simulated_random_comp = data_table[
    (data_table["dataType"] == "simulation") * (data_table["component"] == "random")
]
simulated_community_comp = data_table[
    (data_table["dataType"] == "simulation") * (data_table["component"] == "community")
]
analytical_random_comp = data_table[data_table["dataType"] == "analytical"]

# %% Normalize the density function
analytical_random_comp = analytical_random_comp.sort_values(by="eigs")

analytical_random_comp["prob_mass"] = analytical_random_comp["prob_mass"] / metrics.auc(
    analytical_random_comp["eigs"].values,
    analytical_random_comp["prob_mass"].values,
)
support = analytical_random_comp["eigs"].max()
# ax.plot(zs, density / metrics.auc(zs, density), color="red")
# %%
sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(4.5, 5))

ax = sns.histplot(
    simulated_random_comp, x="eigs", stat="density", common_norm=False, ax=ax
)
sns.lineplot(data=analytical_random_comp, x="eigs", y="prob_mass", color="red", ax=ax)
ax.axvline(support, color="black", ls="--")
ax.legend(frameon=False)
ax.set_xlabel("Eigenvalue")
ax.set_ylabel("Probability")
sns.despine()
fig.savefig(
    output_file,
    bbox_inches="tight",
)
