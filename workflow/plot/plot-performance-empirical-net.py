# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import color_palette as cp
import sys

if "snakemake" in sys.modules:
    result_file = snakemake.input["result_file"]
else:
    result_file = "../../data/empirical/all-result.csv"
data_table = pd.read_csv(result_file)

data_table["clustering"] = data_table["clustering"].fillna("community-detection")

# Prepare data for plotting
focal_clustering = ["silhouette"]

netdata_list = [
    "polblog",
    "airport",
    "cora",
    "football",
    "polbooks",
    "highschool",
]

eval_metric = "esim"

focal_methods = [
    "node2vec",
    "deepwalk",
    "line",
    "nonbacktracking",
    "leigenmap",
    "modspec",
    "modspec2",
    "flatsbm",
]
df = data_table.copy()

dg0 = df.query("(clustering == 'community-detection')")
dg1 = df.query("(clustering != 'community-detection')")

dflist = [dg1]
for clus in focal_clustering:
    dg0_ = dg0.copy()
    dg0_["clustering"] = clus
    dflist.append(dg0_)

df = pd.concat(dflist)

df = df.query(
    f"((clustering == 'community-detection') or (clustering == @focal_clustering)) and (score_type=='{eval_metric}') and ((normalize==False and dimThreshold==False) or normalize.isnull())"
)

df = df.query("name in @focal_methods")
df = df.query("netdata in @netdata_list")

model_list = cp.get_model_order()
data_model_list = df["name"].unique().tolist()
model_list = [k for k in model_list if k in data_model_list]

df["name"] = pd.Categorical(df["name"], categories=model_list, ordered=True)

model_color = cp.get_model_colors()
model_markers = cp.get_model_markers()
model_marker_size = cp.get_model_marker_size()
model_linestyles = cp.get_model_linestyles()
model_names = cp.get_model_names()
model_edge_color = cp.get_model_edge_colors()
model_groups = cp.get_model_groups()
model_color

# Rename the model names
df["name"] = df["name"].map(model_names)


# %% Plotting
sns.set_style("white")
sns.set(font_scale=1.4)
sns.set_style("ticks")


fig, axes = plt.subplots(figsize=(20, 10), nrows=2, ncols=len(netdata_list) // 2)


for i, (ax, netdata) in enumerate(zip(axes.flatten(), netdata_list)):
    df_ = df.query("netdata == @netdata")
    sns.boxplot(
        data=df_,
        x="name",
        y="score",
        order=[model_names[c] for c in focal_methods],
        color="#fdfdfd",
        ax=ax,
    )
    sns.stripplot(
        data=df_,
        x="name",
        y="score",
        order=[model_names[c] for c in focal_methods],
        ax=ax,
        palette={model_names[k]: v for k, v in model_color.items()},
        edgecolor="k",
        linewidth=1,
        s=8,
        alpha=0.8,
    )
    ax.set_title(netdata)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if i == 0:
        ax.legend().remove()

    if i < 3:
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

fig.savefig(
    f"empirical-performance~{focal_clustering[0]}.pdf", bbox_inches="tight", dpi=300
)
