# -*- coding: utf-8 -*-
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import color_palette as cp

# -----------------------------------------------
# Parameters
# -----------------------------------------------
if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    params["name"] = snakemake.params["model_names"]
    params["dimThreshold"] = False
    params["normalize"] = False
    title = (
        snakemake.params["title"]
        if "title" in snakemake.params.keys()
        else None
    )
    with_legend = (
        str(snakemake.params["with_legend"]) == "True"
        if "with_legend" in snakemake.params.keys()
        else "True"
    )
else:
    input_file = "../../data/multi_partition_model/all-result.csv"
    output_file = "../data/"
    title = None
    with_legend = True
    params = {
        "q": 50,
        "dim": 16,
        "n": 10000,
        "metric": "cosine",
        "length": 10,
        "name": [
            "node2vec",
            "deepwalk",
            "line",
            "nonbacktracking",
            "infomap",
            "flatsbm",
            "modspec",
            "modspec2",
            "leigenmap",
            "bp",
        ],
        "score_type": "esim",
        "cave": 10,
        "dimThreshold": False,
        "normalize": True,
    }

# -----------------------------------------------
# Load and filter data
# -----------------------------------------------
data_table = pd.read_csv(input_file)

plot_data = data_table.copy()
for col, values in params.items():
    if col not in plot_data.columns:
        continue
    if not isinstance(values, list):
        values = [values]
    plot_data = plot_data[(plot_data[col].isin(values)) | pd.isna(plot_data[col])]

plot_data = plot_data[plot_data["name"] != "levy-word2vec"]

# -----------------------------------------------
# Plot styling
# -----------------------------------------------
sns.set_style("white")
sns.set(font_scale=2.0)
sns.set_style("ticks")

model_order = cp.get_model_order()
available_models = plot_data["name"].unique().tolist()
model_order = [m for m in model_order if m in available_models]

model_colors = cp.get_model_colors()
model_markers = cp.get_model_markers()
model_marker_sizes = cp.get_model_marker_size()
model_linestyles = cp.get_model_linestyles()
model_display_names = cp.get_model_names()
model_edge_colors = cp.get_model_edge_colors()
model_groups = cp.get_model_groups()

# -----------------------------------------------
# Draw lines (plot in reverse order so first items render on top)
# -----------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
for name in model_order[::-1]:
    subset = plot_data[plot_data["name"] == name]
    color = model_colors[name]

    if color == "white":
        sns.lineplot(
            data=subset,
            x="mu",
            y="score",
            dashes=model_linestyles[name],
            color="black",
            ax=ax,
        )

    sns.lineplot(
        data=subset,
        x="mu",
        y="score",
        marker=model_markers[name],
        dashes=model_linestyles[name],
        color=color,
        markeredgecolor=model_edge_colors[name],
        markersize=model_marker_sizes[name],
        label=name,
        ax=ax,
    )

ax.plot([0.5], [0.5], marker="None", linestyle="None", label="dummy-tophead")

# -----------------------------------------------
# Axes labels and limits
# -----------------------------------------------
ax.set_xlabel(r"Mixing rate, $\mu$")
if params["score_type"] == "nmi":
    ax.set_ylabel(r"Normalized Mutual Information")
else:
    ax.set_ylabel(r"Element-centric similarity")

mu_max = 1 - 1 / np.sqrt(params["cave"])
ax.axvline(mu_max, color="black", linestyle="--", zorder=1)

ax.set_ylim(-0.03, 1.05)
ax.set_xlim(0, 1.01)
tick_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
tick_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
ax.set_xticks(tick_values)
ax.set_xticklabels(tick_labels)
ax.set_yticks(tick_values)
ax.set_yticklabels(tick_labels)

# -----------------------------------------------
# Legend: keep only models that belong to a known group
# -----------------------------------------------
current_handles, current_labels = ax.get_legend_handles_labels()
legend_handles, legend_labels = [], []
for handle, label in zip(current_handles, current_labels):
    if label not in model_groups:
        continue
    display_name = model_display_names.get(label, label)
    legend_handles.append(handle)
    legend_labels.append(display_name)

if with_legend:
    ax.legend(
        legend_handles[::-1],
        legend_labels[::-1],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fontsize=10,
    )
    ax.set_xlabel("")
else:
    ax.legend().remove()

sns.despine()

if title is not None:
    ax.set_title("")

fig.savefig(
    output_file,
    bbox_inches="tight",
    dpi=300,
)
