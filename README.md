# Network clustering via neural embedding

## Paper
```
```

To cite our work, please use the following BibTeX entry:
```bibtex
@article{neuralemb,
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

Load the network for demonstration
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

```python

# DeepWalk
model = embcom.embeddings.DeepWalk(window_length=window_length, num_walks=num_walks)

# Laplacian EigenMap
model = embcom.embeddings.LaplacianEigenMap()

# Modularity spectral Embedding
model = embcom.embeddings.ModularitySpectralEmbedding()

# Non-backtracking spectral embedding
model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
```

### Belief Propagation Algorithm

We have developed a Python wrapper for the belief propagation method. See [this package](https://github.com/skojaku/BeliefPropagation) for details.

