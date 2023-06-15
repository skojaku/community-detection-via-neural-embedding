# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-04 00:14:13
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-11 21:03:06
# %%
import belief_propagation
import networkx as nx
import numpy as np
from scipy import sparse

# Create a karate club network.
A = nx.to_scipy_sparse_matrix(nx.karate_club_graph())

# Detect communities
# A: scipy sparse matrix. CSR format.
# q: Number of communities
community_ids = belief_propagation.detect(A, q=2, dumping_rate = 0.5)
print(community_ids)
np.savez("result.npz", com=community_ids)

# %%
