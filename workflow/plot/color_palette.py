# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-07-11 22:08:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-05-09 15:12:14
import seaborn as sns
import matplotlib.pyplot as plt


def get_model_order():
    return [
        "node2vec",
        "depthfirst-node2vec",
        "deepwalk",
        "line",
        "torch-modularity",
        "torch-laplacian-eigenmap",
        "nonbacktracking",
        "linearized-node2vec",
        "modspec",
        "leigenmap",
        "bp",
        "infomap",
        "flatsbm",
        "non-backtracking-node2vec",
        "non-backtracking-deepwalk",
        "depthfirst-node2vec",
    ]


def get_model_names():
    return {
        "bp": "Belief Propagation",
        "node2vec": "node2vec",
        "linearized-node2vec": "Spectral node2vec",
        "modspec": "Modularity",
        "leigenmap": "L-EigenMap",
        "torch-modularity": "Neural modularity",
        "torch-laplacian-eigenmap": "Neural L-EigenMap",
        "non-backtracking-node2vec": "Non-backtracking node2vec",
        "non-backtracking-deepwalk": "Non-backtracking DeepWalk",
        "depthfirst-node2vec": "Biased node2vec (p=10,q=0.1)",
        "deepwalk": "DeepWalk",
        "line": "LINE",
        "nonbacktracking": "Non-backtracking",
        "infomap": "Infomap",
        "flatsbm": "Flat SBM",
    }


def get_model_colors():
    cmap = sns.color_palette().as_hex()
    bcmap = sns.color_palette("muted").as_hex()
    mcmap = sns.color_palette("colorblind").as_hex()

    neural_emb_color = bcmap[3]
    spec_emb_color = bcmap[0]
    com_color = bcmap[1]
    neural_emb_color_2 = bcmap[2]
    return {
        "node2vec": "red",
        "deepwalk": sns.desaturate(neural_emb_color, 0.8),
        "line": sns.desaturate(neural_emb_color, 0.2),
        "linearized-node2vec": spec_emb_color,
        "torch-modularity": neural_emb_color_2,
        "torch-laplacian-eigenmap": sns.desaturate(neural_emb_color_2, 0.2),
        "modspec": sns.desaturate(spec_emb_color, 0.8),
        "leigenmap": sns.desaturate(spec_emb_color, 0.2),
        "bp": "k",
        "infomap": "#4d4d4d",
        "flatsbm": "#8d8d8d",
        "nonbacktracking": "blue",
        "non-backtracking-node2vec": "red",
        "non-backtracking-deepwalk": "blue",
        "depthfirst-node2vec": sns.desaturate(neural_emb_color, 0.1),
    }


def get_model_edge_colors():
    return {
        "node2vec": "black",
        "deepwalk": "white",
        "line": "white",
        "torch-modularity": "black",
        "torch-laplacian-eigenmap": "black",
        "linearized-node2vec": "black",
        "modspec": "white",
        "leigenmap": "white",
        "bp": "k",
        "nonbacktracking": "black",
        "non-backtracking-node2vec": "white",
        "non-backtracking-deepwalk": "white",
        "infomap": "white",
        "flatsbm": "white",
        "depthfirst-node2vec": "white",
    }


def get_model_linestyles():
    return {
        "node2vec": (1, 0),
        "deepwalk": (1, 1),
        "line": (2, 2),
        "torch-modularity": (1, 1),
        "torch-laplacian-eigenmap": (2, 2),
        "linearized-node2vec": (1, 0),
        "modspec": (1, 1),
        "leigenmap": (2, 2),
        "nonbacktracking": (1, 0),
        "bp": (1, 0),
        "infomap": (1, 1),
        "flatsbm": (2, 2),
        "depthfirst-node2vec": (1, 3),
    }


def get_model_markers():
    return {
        "node2vec": "s",
        "line": "s",
        "torch-modularity": "D",
        "torch-laplacian-eigenmap": "D",
        "deepwalk": "s",
        "linearized-node2vec": "o",
        "modspec": "o",
        "leigenmap": "o",
        "nonbacktracking": "o",
        "non-backtracking-node2vec": "o",
        "depthfirst-node2vec": "o",
        "non-backtracking-deepwalk": "v",
        "bp": "v",
        "infomap": "v",
        "flatsbm": "v",
    }


def get_model_marker_size():
    return {
        "node2vec": 10,
        "line": 10,
        "torch-modularity": 5,
        "torch-laplacian-eigenmap": 5,
        "deepwalk": 10,
        "linearized-node2vec": 10,
        "modspec": 10,
        "leigenmap": 10,
        "nonbacktracking": 10,
        "non-backtracking-node2vec": 10,
        "depthfirst-node2vec": 10,
        "non-backtracking-deepwalk": 10,
        "bp": 10,
        "infomap": 10,
        "flatsbm": 10,
    }


def get_model_groups():
    return {
        "bp": "community_detection",
        "node2vec": "neural",
        "line": "neural",
        "torch-modularity": "neural",
        "torch-laplacian-eigenmap": "neural",
        "deepwalk": "neural",
        "linearized-node2vec": "spectral",
        "modspec": "spectral",
        "leigenmap": "spectral",
        "nonbacktracking": "spectral",
        "non-backtracking-node2vec": "neural",
        "depthfirst-node2vec": "neural",
        "non-backtracking-deepwalk": "neural",
        "infomap": "community_detection",
        "flatsbm": "community_detection",
    }
