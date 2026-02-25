# -*- coding: utf-8 -*-
import seaborn as sns


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
        "adjspec",
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
        "adjspec": "Adjacency",
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
    palette = sns.color_palette().as_hex()

    base_neural = palette[3]
    base_spectral = palette[0]
    base_neural_alt = palette[2]

    return {
        "node2vec": "red",
        "deepwalk": sns.desaturate(base_neural, 0.8),
        "line": sns.desaturate(base_neural, 0.2),
        "linearized-node2vec": base_spectral,
        "torch-modularity": base_neural_alt,
        "torch-laplacian-eigenmap": sns.desaturate(base_neural_alt, 0.2),
        "adjspec": sns.desaturate(base_spectral, 0.8),
        "modspec": sns.desaturate(base_spectral, 0.2),
        "leigenmap": "#c2c1f1",
        "bp": "k",
        "infomap": "#8d8d8d",
        "flatsbm": "#998248",
        "nonbacktracking": "blue",
        "non-backtracking-node2vec": "red",
        "non-backtracking-deepwalk": "blue",
        "depthfirst-node2vec": sns.desaturate(base_neural, 0.1),
    }


def get_model_edge_colors():
    return {
        "node2vec": "black",
        "deepwalk": "white",
        "line": "white",
        "torch-modularity": "black",
        "torch-laplacian-eigenmap": "black",
        "linearized-node2vec": "black",
        "adjspec": "white",
        "modspec": "k",
        "leigenmap": "k",
        "bp": "k",
        "nonbacktracking": "black",
        "non-backtracking-node2vec": "white",
        "non-backtracking-deepwalk": "white",
        "infomap": "k",
        "flatsbm": "k",
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
        "adjspec": (1, 1),
        "modspec": (1, 2),
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
        "adjspec": "o",
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
        "adjspec": 10,
        "modspec": 10,
        "leigenmap": 10,
        "nonbacktracking": 10,
        "non-backtracking-node2vec": 10,
        "depthfirst-node2vec": 10,
        "non-backtracking-deepwalk": 10,
        "bp": 11,
        "infomap": 11,
        "flatsbm": 11,
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
        "adjspec": "spectral",
        "modspec": "spectral",
        "leigenmap": "spectral",
        "nonbacktracking": "spectral",
        "non-backtracking-node2vec": "neural",
        "depthfirst-node2vec": "neural",
        "non-backtracking-deepwalk": "neural",
        "infomap": "community_detection",
        "flatsbm": "community_detection",
    }
