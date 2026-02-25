"""Plot community detection performance on empirical networks as box/strip plots."""
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import color_palette as cp

# --- I/O configuration ---
if "snakemake" in sys.modules:
    result_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    focal_clustering = snakemake.wildcards.clustering
else:
    result_file = "../../data/empirical/evaluations/all-result.csv"
    focal_clustering = "silhouette"
    output_file = "tmp.pdf"

# --- Load and prepare data ---
df = pd.read_csv(result_file)
df["clustering"] = df["clustering"].fillna("community-detection")

NETWORK_NAMES = [
    "polblog",
    "airport",
    "cora",
    "football",
    "polbooks",
    "highschool",
]

EVAL_METRIC = "esim"

FOCAL_METHODS = [
    "node2vec",
    "deepwalk",
    "line",
    "nonbacktracking",
    "leigenmap",
    "adjspec",
    "modspec",
    "flatsbm",
]

# Keep only rows matching: the focal clustering (or direct community detection),
# the chosen evaluation metric, and non-normalized / non-thresholded results.
df = df.query(
    "(clustering == 'community-detection' or clustering == @focal_clustering)"
    f" and score_type == '{EVAL_METRIC}'"
    " and (normalize == False and dimThreshold == False or normalize.isnull())"
)

df = df.query("name in @FOCAL_METHODS and netdata in @NETWORK_NAMES")

# Sort models by the canonical display order from color_palette
all_model_order = cp.get_model_order()
available_models = df["name"].unique().tolist()
model_order = [m for m in all_model_order if m in available_models]

df["name"] = pd.Categorical(df["name"], categories=model_order, ordered=True)

model_colors = cp.get_model_colors()
model_names = cp.get_model_names()

# Map internal model keys to human-readable display names
df["name"] = df["name"].map(model_names)
display_name_order = [model_names[m] for m in FOCAL_METHODS]
display_color_palette = {model_names[k]: v for k, v in model_colors.items()}

# --- Plotting ---
sns.set_style("white")
sns.set(font_scale=1.4)
sns.set_style("ticks")

n_cols = len(NETWORK_NAMES) // 2
fig, axes = plt.subplots(figsize=(20, 10), nrows=2, ncols=n_cols)

for i, (ax, network) in enumerate(zip(axes.flatten(), NETWORK_NAMES)):
    network_df = df.query("netdata == @network")

    available_in_network = set(network_df["name"].dropna().unique())
    network_order = [m for m in display_name_order if m in available_in_network]

    sns.boxplot(
        data=network_df,
        x="name",
        y="score",
        order=network_order,
        color="#fdfdfd",
        ax=ax,
    )
    sns.stripplot(
        data=network_df,
        x="name",
        y="score",
        order=network_order,
        ax=ax,
        palette=display_color_palette,
        edgecolor="k",
        linewidth=1,
        s=8,
        alpha=0.8,
    )

    ax.set_title(network)
    ax.set_xlabel("")
    ax.set_ylabel("")

    if i == 0:
        ax.legend().remove()

    # Only show x-axis labels on the bottom row
    if i < n_cols:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            wrap=True,
        )

sns.despine()
fig.savefig(output_file, bbox_inches="tight", dpi=300)
