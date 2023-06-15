# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-06-15 12:30:50
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-15 12:50:47
import lfr

params = {
    "N": 1000,  # number of nodes
    "k": 50,  # average degree
    "maxk": 100,  # maximum degree
    "minc": 20,  # minimum community size
    "maxc": 100,  # maximum community size
    "tau": 3.0,  # degree exponent
    "tau2": 2.0,  # community size exponent
    "mu": 0.5,  # Mixing rate
}


ng = lfr.NetworkGenerator()
data = ng.generate(**params)

net = data["net"]  # scipy.csr_sparse matrix
community_table = data["community_table"]  # pandas DataFrame
seed = data["seed"]  # Seed value
