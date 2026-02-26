"""Plot community detection performance vs mixing rate (mu) for LFR benchmarks."""
import sys
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import color_palette as cp

# -------------------------------------------------------------------------
# Parameters: from Snakemake or defaults for standalone execution
# -------------------------------------------------------------------------
if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    output_file = snakemake.output["output_file"]
    params = snakemake.params["parameters"]
    params["name"] = snakemake.params["model_names"]
    title = snakemake.params.get("title", None)
    with_legend = str(snakemake.params.get("with_legend", "True")) == "True"
else:
    input_file = "../../data/lfr/all-result.csv"
    output_file = "../data/"
    title = None
    with_legend = True
    params = {
        "dim": 128,
        "n": 10000,
        "metric": "cosine",
        "length": 10,
        "k": 10,
        "tau": 3.0,
        "clustering": "kmeans",
        "score_type": "esim",
        "dimThreshold": True,
        "name": [
            "node2vec",
            "deepwalk",
            "line",
            "nonbacktracking",
            "infomap",
            "flatsbm",
            "modspec",
            "leigenmap",
            "bp",
        ],
        "normalize": True,
    }

# -------------------------------------------------------------------------
# Load and filter data
# -------------------------------------------------------------------------
data_table = pd.read_csv(input_file)

plot_data = data_table.copy()
for col, values in params.items():
    if col not in plot_data.columns:
        continue
    if not isinstance(values, list):
        values = [values]
    plot_data = plot_data[plot_data[col].isin(values) | pd.isna(plot_data[col])]

plot_data = plot_data[plot_data["name"] != "levy-word2vec"]

# -------------------------------------------------------------------------
# Visual style lookups from color_palette
# -------------------------------------------------------------------------
model_colors = cp.get_model_colors()
model_markers = cp.get_model_markers()
model_marker_sizes = cp.get_model_marker_size()
model_linestyles = cp.get_model_linestyles()
model_display_names = cp.get_model_names()
model_edge_colors = cp.get_model_edge_colors()
model_groups = cp.get_model_groups()

# Keep only models present in the data, in the canonical order
ordered_models = [m for m in cp.get_model_order() if m in plot_data["name"].unique()]

# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------
sns.set_style("white")
sns.set(font_scale=2.0)
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(6, 5))

# Draw models in reverse order so first items appear on top
for model in reversed(ordered_models):
    subset = plot_data[plot_data["name"] == model]
    color = model_colors[model]
    edge_color = model_edge_colors[model]

    # White-filled markers need a black line underneath for visibility
    if color == "white":
        sns.lineplot(
            data=subset,
            x="mu",
            y="score",
            dashes=model_linestyles[model],
            color="black",
            ax=ax,
        )

    sns.lineplot(
        data=subset,
        x="mu",
        y="score",
        marker=model_markers[model],
        dashes=model_linestyles[model],
        color=color,
        markeredgecolor=edge_color,
        markersize=model_marker_sizes[model],
        label=model,
        ax=ax,
    )

# Invisible dummy entry used to insert a heading row in the legend
ax.plot([0.5], [0.5], marker="None", linestyle="None", label="dummy-tophead")

# Axis labels
ax.set_xlabel(r"Mixing rate, $\mu$")

score_labels = {
    "nmi": r"Normalized Mutual Information",
    "esim": r"Element-centric similarity",
}
score_type = params["score_type"]
if score_type in score_labels:
    ax.set_ylabel(score_labels[score_type])

# Axis limits and ticks
ax.set_ylim(-0.03, 1.05)
ax.set_xlim(0, 1.01)
tick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
tick_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
ax.set_xticks(tick_values)
ax.set_xticklabels(tick_labels)
ax.set_yticks(tick_values)
ax.set_yticklabels(tick_labels)

# -------------------------------------------------------------------------
# Legend: only show models that belong to a known group
# -------------------------------------------------------------------------
handles, labels = ax.get_legend_handles_labels()
legend_handles = []
legend_labels = []
for handle, label in zip(handles, labels):
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
        fontsize=12,
    )
    ax.set_xlabel("")
else:
    ax.legend().remove()

sns.despine()

if title is not None:
    ax.set_title(textwrap.fill(title, width=42))

fig.savefig(output_file, bbox_inches="tight", dpi=300)
