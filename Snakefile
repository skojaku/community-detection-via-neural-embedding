import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace

configfile: "workflow/config.yaml"
include: "./utils.smk"

# ==========
# Parameters
# ==========

#
# Embedding parameters
#
emb_params = {
    "model_name": [
        "node2vec",
        "deepwalk",
        "line",
        "leigenmap",
        "modspec",
        "modspec2",
        "nonbacktracking",
    ],
    "window_length": [10],
    "dim": [16, 64, 128],
}

#
# Community detection parameters
#
com_detect_params = {
    "model_name": ["infomap", "flatsbm", "bp"],
}

#
# Data clustering parameters
#
clustering_params = {
    "metric": ["cosine"],
    "clustering": ["voronoi", "kmeans", "silhouette"],
}

#
# Number of samples
#
N_SAMPLES = 1

#
# Parmaters for the planted Partition models
#
net_params = {
    "n": [10000, 100000],  # Network size
    "q": [2, 50],  # Number of communities
    "cave": [5, 10, 50],  # average degree
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    "sample": np.arange(N_SAMPLES),  # Number of samples
}

#
# Parmaters for the LFR benchmark
#
lfr_net_params = {
    "n": [10000],  # Network size
    "k": [5, 10, 50],  # Average degree
    "tau": [3],  # degree exponent
    "tau2": [1],  # community size exponent
    "minc": [50],  # min community size
    "mu": ["%.2f" % d for d in np.linspace(0.1, 1, 19)],
    "sample": np.arange(N_SAMPLES),  # Number of samples
}

fig_params_perf_vs_mixing = {
    "data": ["multi_partition_model"],
    "n": net_params["n"],
    "q": net_params["q"],
    "cave": net_params["cave"],
    "dim": emb_params["dim"],
    "metric": ["cosine"],
    "length": emb_params["window_length"],
    "clustering": clustering_params["clustering"],
    "score_type": ["esim", "nmi"],
}

fig_lfr_params_perf_vs_mixing = {
    "data": ["lfr"],
    "n": lfr_net_params["n"],
    "k": lfr_net_params["k"],  # Average degree
    "tau":lfr_net_params["tau"],
    "length": emb_params["window_length"],
    "dim": emb_params["dim"],
    "metric": ["cosine"],
    "clustering": clustering_params["clustering"],
    "score_type": ["esim", "nmi"],
}

# =============================
# Folders
# =============================

DATA_DIR = config["data_dir"]
FIG_DIR = j("figs", "{data}")
NET_DIR = j(DATA_DIR, "{data}", "networks")
EMB_DIR = j(DATA_DIR, "{data}", "embedding")
COM_DIR = j(DATA_DIR, "{data}", "communities")
EVA_DIR = j(DATA_DIR, "{data}", "evaluations")

# All results
EVAL_CONCAT_FILE = j(EVA_DIR, f"all-result.csv")

# ============
# Data specific
# ============

FIG_PERFORMANCE_VS_MIXING_ALL = j(FIG_DIR, "all_perf_vs_mixing.pdf",)


include: "./Snakefile_multipartition_files.smk"


include: "./Snakefile_lfr_files.smk"


include: "./Snakefile_empirical.smk"


# ======
# RULES
# ======

DATA_LIST = ["multi_partition_model", "lfr", "empirical"]


rule all:
    input:
        #
        # Multipartition
        #
        expand(
            EVAL_EMB_FILE,
            data="multi_partition_model",
            **net_params,
            **emb_params,
            **clustering_params
        ),
        expand(EMB_FILE, data="multi_partition_model", **net_params, **emb_params),
        #
        # LFR
        #
        expand(
            LFR_EVAL_EMB_FILE,
            data="lfr",
            **lfr_net_params,
            **emb_params,
            **clustering_params
        ),
        expand(LFR_EMB_FILE, data="lfr", **lfr_net_params, **emb_params),
        #
        # Empirical networks
        #
        expand(NET_EMP_FILE,  **net_emp_params),
        expand(EMB_EMP_FILE, **net_emp_params, **emb_params, sample = range(N_SAMPLES_EMP)),
        expand(EVAL_EMB_EMP_FILE, **net_emp_params, **emb_params, **clustering_params, sample = range(N_SAMPLES_EMP)),
        expand(EVAL_EMP_FILE, **net_emp_params, **clustering_params, **com_detect_params, sample = range(N_SAMPLES_EMP))


rule figs:
    input:
        expand(FIG_PERFORMANCE_VS_MIXING, **fig_params_perf_vs_mixing),
        expand(FIG_LFR_PERFORMANCE_VS_MIXING, **fig_lfr_params_perf_vs_mixing),
        expand(FIG_PERFORMANCE_VS_MIXING_ALL, data=DATA_LIST),

rule _all:
    input:
        expand(FIG_EMP_PERFORMANCE, data="empirical", clustering=clustering_params["clustering"])
