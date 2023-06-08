# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-07-11 22:08:07
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-25 16:11:17
# %%
import sys
import textwrap

import color_palette as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    model_names = snakemake.params["model_names"]
    params["name"] = model_names
    title = (
        snakemake.params["title"] if "title" in list(snakemake.params.keys()) else None
    )
    with_legend = (
        str(snakemake.params["with_legend"]) == "True"
        if "with_legend" in list(snakemake.params.keys())
        else "True"
    )
else:
    input_file = "../../data/lfr/all-result.csv"
    output_file = "../data/"
    with_legend = True
    params = {
        "dim": 64,
        "n": 10000,
        "metric": "cosine",
        "length": 10,
        "k": 10,
        "tau": 2.1,
        "clustering": "voronoi",
        "score_type": "esim",
        "dimThreshold": True,
        "name": [
            "node2vec",
            "deepwalk",
            "line",
            # "linearized-node2vec",
            "nonbacktracking",
            "infomap",
            "flatsbm",
            "modspec",
            "leigenmap",
            "bp",
        ],
        "normalize": True,
    }

#
# Load
#
data_table = pd.read_csv(input_file)

#
plot_data = data_table.copy()
for k, v in params.items():
    if k not in plot_data.columns:
        continue
    if not isinstance(v, list):
        v = [v]
    plot_data = plot_data[(plot_data[k].isin(v)) | pd.isna(plot_data[k])]

plot_data = plot_data[plot_data["name"] != "levy-word2vec"]

# %%
plot_data["name"].unique()

# %%
#
# Plot
#
sns.set_style("white")
sns.set(font_scale=2.0)
sns.set_style("ticks")

model_list = cp.get_model_order()
data_model_list = plot_data["name"].unique().tolist()
model_list = [k for k in model_list if k in data_model_list]

model_color = cp.get_model_colors()
model_markers = cp.get_model_markers()
model_marker_size = cp.get_model_marker_size()
model_linestyles = cp.get_model_linestyles()
model_names = cp.get_model_names()
model_edge_color = cp.get_model_edge_colors()
model_groups = cp.get_model_groups()

fig, ax = plt.subplots(figsize=(6, 5))

for name in model_list[::-1]:
    color = model_color[name]
    markeredgecolor = model_edge_color[name]
    if color == "white":
        ax = sns.lineplot(
            data=plot_data[plot_data["name"] == name],
            x="mu",
            y="score",
            dashes=model_linestyles[name],
            color="black",
            ax=ax,
        )

    ax = sns.lineplot(
        data=plot_data[plot_data["name"] == name],
        x="mu",
        y="score",
        marker=model_markers[name],
        dashes=model_linestyles[name],
        color=color,
        markeredgecolor=markeredgecolor,
        markersize=model_marker_size[name],
        label=name,
        ax=ax,
    )
(dummy,) = ax.plot([0.5], [0.5], marker="None", linestyle="None", label="dummy-tophead")
# ax = sns.lineplot(
#    data=plot_data,
#    x="mu",
#    y="score",
#    hue="name",
#    style="name",
#    markers=model_markers,
#    style_order=model_list,
#    hue_order=model_list,
#    palette=model_color,
#    markeredgecolor="k",
#    ax=ax,
# )

ax.set_xlabel(r"Mixing rate, $\mu$")

if params["score_type"] == "nmi":
    ax.set_ylabel(r"Normalized Mutual Information")
elif params["score_type"] == "esim":
    ax.set_ylabel(r"Element-centric similarity")

ax.set_ylim(-0.03, 1.05)
ax.set_xlim(0, 1.01)
xtick_loc = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
xtick_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
ax.set_xticks(xtick_loc)
ax.set_xticklabels(xtick_labels)
ax.set_yticks(xtick_loc)
ax.set_yticklabels(xtick_labels)
# mu_max = 1 - 1 / np.sqrt(params["cave"])
# ax.axvline(mu_max, color="black", linestyle="--")

current_handles, current_labels = ax.get_legend_handles_labels()
new_handles = []
new_labels = []
for i, l in enumerate(current_labels):
    if l not in model_groups:
        continue
    new_handles.append(current_handles[i])
    new_labels.append(model_names[l] if l in model_names else l)

if with_legend:
    lgd = ax.legend(
        new_handles[::-1],
        new_labels[::-1],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fontsize=12,
    )
    ax.set_xlabel("")
else:
    ax.legend().remove()
sns.despine()

if title is not None:
    ax.set_title(textwrap.fill(title, width=42))

fig.savefig(
    output_file,
    bbox_inches="tight",
    dpi=300,
)

# %%
