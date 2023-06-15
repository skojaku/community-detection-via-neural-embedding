# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-08 16:38:19
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-15 12:04:07
import unittest
import networkx as nx
import numpy as np
from scipy import sparse
import embcom


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_mod_spectral(self):
        model = embcom.embeddings.ModularitySpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=12)

    def test_adj_spectral(self):
        model = embcom.embeddings.AdjacencySpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=12)

    def test_leigenmap(self):
        model = embcom.embeddings.LaplacianEigenMap()
        model.fit(self.A)
        vec = model.transform(dim=12)

    def test_node2vec(self):
        model = embcom.embeddings.Node2Vec()
        model.fit(self.A)
        vec = model.transform(dim=12)

    def test_deepwalk(self):
        model = embcom.embeddings.DeepWalk()
        model.fit(self.A)
        vec = model.transform(dim=12)

    def test_nonbacktracking(self):
        model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
        model.fit(self.A)
        vec = model.transform(dim=12)

    def test_normalized_trans_matrix_spec_embedding(self):
        model = embcom.embeddings.LinearizedNode2Vec()
        model.fit(self.A)
        vec = model.transform(dim=12)


if __name__ == "__main__":
    unittest.main()
