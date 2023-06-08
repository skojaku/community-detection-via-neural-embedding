# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-12-12 15:39:03
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-24 14:48:55
# =========================================
# This notebook will visualize the loss landscape of the original and regularized loss.
# The regularization is attributed to the stochastic gradient descent algorithm.
# See:
# Smith, Samuel L., Benoit Dherin, David Barrett, and Soham De. “On the Origin of Implicit Regularization in Stochastic Gradient Descent,” 2022.
# https://openreview.net/forum?id=rq_Qr0c1Hyo
# =========================================
# %%
# Import libraries
#
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import sparse, stats
from tqdm.auto import tqdm
import embcom

# ==============
# Parameters
# ==============
sgd_epsilon = 5e-3  # Learning rate for SGD
n_batches = 2000  # Number of batches per iter
n_iter = 1
n_bins = 40

# ==============
# Loading
# ==============
# Get a network generated with the SBM.
# %%
if "snakemake" in sys.modules:
    node_table_file = snakemake.input["node_table_file"]
    net_file = snakemake.input["net_file"]
    n_nodes = int(snakemake.params["n"])
    mu = float(snakemake.params["mu"])
    cave = float(snakemake.params["cave"])
    K = int(snakemake.params["K"])
    output_file = snakemake.output["output_file"]
else:
    n_nodes = 10000  # number of nodes
    mu = 0.2  # Mixing rate
    cave = 10  # Average degree
    K = 50  # Number of communities
    node_table_file = f"../../data/multi_partition_model/networks/node_n~{n_nodes}_K~{K}_cave~{cave}_mu~{mu:.2f}_sample~2.npz"
    net_file = f"../../data/multi_partition_model/networks/net_n~{n_nodes}_K~{K}_cave~{cave}_mu~{mu:.2f}_sample~2.npz"
    output_file = "landscape_modularity_loss.pdf"

net = sparse.load_npz(net_file)
node_table = pd.read_csv(node_table_file)
memberships = node_table["membership"].values

# ==============
# Embedding
# ==============
# Modularity embedding
emb = (
    embcom.ModularitySpectralEmbedding(reconstruction_vector=True, p=100, q=100)
    .fit(net)
    .transform(dim=K)
)
emb[np.isnan(emb)] = 0
emb[np.isinf(emb)] = 0

# Ground-truth
q = 1 / K
cout = cave * mu
cin = (cave / q) * (1 - (1 - q) * mu)
pin = cin / n_nodes
pout = cout / n_nodes
pave = cave / n_nodes
S = np.ones((K, K)) * pout + np.eye(K) * (pin - pout) - pave
lam, s = np.linalg.eig(S)
U = sparse.csr_matrix(
    (np.ones_like(memberships), (np.arange(n_nodes), memberships)), shape=(n_nodes, K)
)
emb_true = U @ s
emb_true = np.einsum(
    "ij,j->ij",
    emb_true,
    1 / np.maximum(1e-32, np.array(np.linalg.norm(emb_true, axis=0)).reshape(-1)),
)
emb_true = emb_true @ np.diag(np.sqrt(np.maximum(0, lam * (n_nodes / K))))
emb_true = np.real(emb_true)

# =====================
# Loss evaluation
# =====================

# Original Modularity Loss
def modularity_loss(A, X):
    deg = np.array(A.sum(axis=0)).reshape(-1)
    double_m = np.sum(deg)
    n_nodes = A.shape[0]
    Qsquared = (
        double_m
        - 2 * np.sum(deg.reshape((1, -1)) @ net @ deg.reshape((-1, 1))) / double_m
        + np.sum(np.power(deg, 2)) ** 2 / double_m**2
    )
    q = -2 * np.trace(X.T @ A @ X - (X.T @ deg) @ (deg.T @ X) / double_m)
    XX = X.T @ X
    q += np.trace(XX @ XX)
    q += Qsquared
    q /= double_m**2
    return q


# Regularized modularity loss
def regularized_modularity_loss(A, X, epsilon, n_batches, n_iter=10):
    def to_mean_matrix(r, c, n, nc=None, mode="all"):
        if nc is None:
            nc = n
        U = sparse.csr_matrix((np.ones_like(r, dtype="float"), (r, c)), shape=(n, nc))

        if mode == "row":
            denom = np.array(U.sum(axis=1)).reshape(-1)
            U = sparse.diags(1 / np.maximum(1, denom)) @ U
        elif mode == "all":
            U.data = U.data / np.sum(U.data)
        elif mode == "none":
            pass
        return U

    deg = np.array(A.sum(axis=0)).reshape(-1)
    double_m = np.sum(deg)
    n_nodes = A.shape[0]
    dave = double_m / n_nodes

    reg = 0
    for _ in range(n_iter):
        batches = generate_modularity_batches(net, n_batches)
        for (
            center,
            context,
            random_center,
            random_context,
            base_center,
            base_context,
        ) in batches:
            n = len(center)
            U = to_mean_matrix(center, context, n_nodes, mode="all")
            Ur = to_mean_matrix(random_center, random_context, n_nodes, mode="all")
            dX = -2 * U @ X + 2 * Ur @ X

            Ub = to_mean_matrix(
                base_center, np.arange(len(center)), n_nodes, len(center), mode="all"
            )
            dX2 = np.einsum(
                "i,ij->ij",
                np.array(
                    np.sum(X[base_center, :] * X[base_context, :], axis=1)
                ).reshape(-1),
                X[base_context, :],
            )
            dX += 2 * (1 / dave) ** 2 * Ub @ dX2

            reg += np.linalg.norm(dX) ** 2
    reg /= n_batches * n_iter
    Q = modularity_loss(A, X)
    return Q + (epsilon / 4) * reg


# Generate batches for modularity loss
def generate_modularity_batches(A, n_batches):
    n_nodes = A.shape[0]
    center, context, _ = sparse.find(A)
    random_center, random_context = np.random.choice(
        center, size=len(center)
    ), np.random.choice(center, size=len(center))
    base_center, base_context = np.random.choice(
        n_nodes, size=len(center)
    ), np.random.choice(n_nodes, size=len(center))

    arrays = [center, context, random_center, random_context, base_center, base_context]

    batches = []
    for ids in np.array_split(
        np.random.choice(n_nodes, size=n_nodes, replace=False), n_batches
    ):
        batches.append([arrays[i][ids].copy() for i in range(len(arrays))])
    return batches


# Modualrity loss for individual batches
def modularity_loss_batch(A, X, batches):
    deg = np.array(A.sum(axis=0)).reshape(-1)
    double_m = np.sum(deg)
    n_nodes = A.shape[0]
    dave = double_m / n_nodes

    Qsquared = (
        double_m
        - 2 * np.sum(deg.reshape((1, -1)) @ net @ deg.reshape((-1, 1))) / double_m
        + np.sum(np.power(deg, 2)) ** 2 / double_m**2
    )
    Qsquared /= double_m**2
    qlist = []
    for (
        center,
        context,
        random_center,
        random_context,
        base_center,
        base_context,
    ) in batches:
        q = -2 * np.mean(np.sum(X[center, :] * X[context, :], axis=1)) / double_m
        q += (
            2
            * np.mean(np.sum(X[random_center, :] * X[random_context, :], axis=1))
            / double_m
        )
        q += (1.0 / dave**2) * np.mean(
            np.sum(X[base_center, :] * X[base_context, :], axis=1) ** 2
        )
        q += Qsquared
        qlist.append(q)
    return np.array(qlist)


# ==========================
# Walking the loss landscape
# ==========================

# Direction
emb_axis = emb_true - emb

# Calculating the regularized and non-regularized modularity loss
results = []
for xt in tqdm(np.linspace(-1, 2, n_bins)):
    emb_t = emb + xt * emb_axis
    qt = modularity_loss(net, emb_t)
    qt_reg = regularized_modularity_loss(
        net, emb_t, epsilon=sgd_epsilon, n_batches=n_batches, n_iter=n_iter
    )
    results.append({"x": xt, "q": qt, "q_reg": qt_reg, "norm": np.linalg.norm(emb_t)})
data_table = pd.DataFrame(results)

# Calculating the non-regularized modularity loss with individual batches
results = []
batches = generate_modularity_batches(net, n_batches=n_batches)
for xt in tqdm(np.linspace(-1, 2, n_bins)):
    emb_t = emb + xt * emb_axis
    qts = modularity_loss_batch(net, emb_t, batches)
    results.append(pd.DataFrame({"x": xt, "q": qts, "batch_id": np.arange(len(qts))}))
ref_data_table = pd.concat(results)

# %%
# ============================
# Plot
# ============================
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
sns.set(font_scale=1.2)
sns.set_style("ticks")

# Canvas
fig, ax = plt.subplots(figsize=(6, 5))

# Set colors
col = "grey"  # sns.color_palette()[0]
reg_col = sns.color_palette("bright")[3]
bcol = sns.color_palette("bright")[0]

# Range
qmax = data_table["q"].max()
qmin = data_table["q"].min()
dq = qmax - qmin

# Plot the original loss landscape
ax = sns.lineplot(
    data=data_table, x="x", y="q", color=bcol, lw=2, label="Original", ax=ax
)
# Plot the regularized loss landscape
ax = sns.lineplot(
    data=data_table,
    x="x",
    y="q_reg",
    color=reg_col,
    ls="-.",
    lw=2,
    label="Regularized",
    ax=ax,
)

# Fill
ax.fill_between(
    data_table["x"], data_table["q"], data_table["q_reg"], color=col, alpha=0.5
)

# Place points to indicate the minimizer
idx_min_q_emp = np.argmin(data_table["q"])
idx_min_q_reg = np.argmin(data_table["q_reg"])
ax.scatter(
    [data_table["x"][idx_min_q_emp]],
    [data_table["q"][idx_min_q_emp]],
    color="white",
    marker="o",
    edgecolor="black",
    s=120,
    zorder=999,
)
ax.scatter(
    [data_table["x"][idx_min_q_reg]],
    [data_table["q_reg"][idx_min_q_reg]],
    color="red",
    edgecolor="black",
    marker="o",
    s=150,
    zorder=999,
)

# Indicate the minimizer and ground-truth
ax.axvline(0, ls=":", color="grey", lw=1)
ax.axvline(1, ls=":", color="grey", lw=1)

# Annotate the minimizer and ground-truth
t = ax.annotate(
    text="Ground-truth", xy=(1, qmax), va="center", ha="center", xycoords="data"
)
t.set_bbox(dict(facecolor="white", alpha=1, edgecolor="white"))
t = ax.annotate(
    text="Minimizer", xy=(0, qmax), va="center", ha="center", xycoords="data"
)
t.set_bbox(dict(facecolor="white", alpha=1, edgecolor="white"))

# Loss-landscape of individual batches
for i, (_, df) in enumerate(ref_data_table.groupby("batch_id")):
    if i > 100:
        continue
    sns.lineplot(
        data=df, x="x", y="q", color="grey", alpha=0.3, lw=0.5, zorder=1, ax=ax
    )

# Add the legend
ax.legend(
    facecolor="white",
    edgecolor="#ffffff",
    loc="best",
    # bbox_to_anchor=(0.55, 0.3, 0.5, 0.5),
    fontsize=12,
)

# Modify ranges
ax.set_xlim(-1, 2)
ax.set_ylim(qmin - 0.3 * dq, qmax)
ax.set_ylabel(r"Loss")
ax.set_xlabel("$\gamma$")
sns.despine()

fig.savefig(output_file, bbox_inches="tight", dpi=300)

# %%
