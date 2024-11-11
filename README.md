# Network community detection via neural embedding

## Paper
- [Nature Communication](https://www.nature.com/articles/s41467-024-52355-w)
- [arXiv version](https://arxiv.org/abs/2306.13400)

```
Sadamori Kojaku, Filippo Radicchi, Yong-Yeol Ahn, Santo Fortunato. "Network community detection via neural embeddings", arXiv: 2306.13400(https://arxiv.org/abs/2306.13400)
```

To cite our work, please use the following BibTeX entry:
```bibtex
@article{kojaku2024network,
  title={Network community detection via neural embeddings},
  author={Kojaku, Sadamori and Radicchi, Filippo and Ahn, Yong-Yeol and Fortunato, Santo},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={9446},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Reproducing Our Results

### Setup

1. Set up the virtual environment and install the required packages:
```bash
conda create -n neuralemb python=3.9
conda activate neuralemb
conda install -c conda-forge mamba -y
mamba install -y -c bioconda -c conda-forge snakemake -y
mamba install -c conda-forge graph-tool scikit-learn numpy==1.23.5 numba scipy pandas networkx seaborn matplotlib gensim ipykernel tqdm black -y
```

2. Install the in-house packages

```bash
cd libs/BeliefPropagation && python3 setup.py build && pip install -e .
cd libs/LFR-benchmark && python3 setup.py build && pip install -e .
cd libs/embcom && pip install -e .
```

4. Create a file `config.yaml` with the following content and place it under the `workflow` folder:
```yaml
data_dir: "data/"
```

Note that the script will generate over 1T byte of data under this `data/` folder. Make sure you have sufficient disk space.

### Run Simulation

Run the following command to execute the `Snakemake` workflow:
```bash
snakemake --cores 24 all
```
This will generate all files needed to produce the figures. Then, run
```bash
snakemake --cores 24 figs
```
You can change the number of cores to use, instead of 24.

## About the Code

### Graph Embedding

We provide a package for graph embedding methods, including node2vec, DeepWalk, LINE, and some conventional graph embedding. The package can be installed using the following command:
```bash
cd libs/embcom && pip install -e .
```

#### Usage
```python
import embcom
import networkx as nx

# Load the network for demonstration
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)

# Loading the node2vec model
# window_length
# : Size of window, T
# num_walks
# : Number of walks
model = embcom.embeddings.Node2Vec(window_length=80, num_walks=20)

# Train
# net
# : scipy sparse matrix
model.fit(net)

# Generate an embedding
# dim
# : Integer
emb = model.transform(dim=dim)
```

Other embedding models:

| Class Name                        | Description                                      |
|-----------------------------------|--------------------------------------------------|
| `embcom.embeddings.Node2Vec`                          | Node2Vec embedding                               |
| `embcom.embeddings.DeepWalk`                          | DeepWalk embedding                               |
| `embcom.embeddings.LINE`                              | LINE embedding                                   |
| `embcom.embeddings.GloVe`                             | GloVe embedding                                  |
| `embcom.embeddings.LaplacianEigenMap`                 | Laplacian EigenMap embedding                     |
| `embcom.embeddings.AdjacencySpectralEmbedding`        | Adjacency Spectral Embedding                     |
| `embcom.embeddings.ModularitySpectralEmbedding`       | Modularity Spectral Embedding                    |
| `embcom.embeddings.NonBacktrackingSpectralEmbedding`  | Non-Backtracking Spectral Embedding              |
| `embcom.embeddings.LinearizedNode2Vec`                | Linearized Node2Vec embedding                    |


### Belief Propagation Algorithm

We have developed a Python wrapper for the belief propagation method. See [this package](https://github.com/skojaku/BeliefPropagation) for details.


### LFR benchmark

We used the original code for the LFR network models,w ith [our Python wrapper](https://github.com/skojaku/LFR-benchmark).
