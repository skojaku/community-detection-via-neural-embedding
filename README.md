# Network clustering via neural embedding 


# Paper 

```
```

How to cite:
```
@article{neuralemb,
}
```

# Repducing our results

## Setup

Set up the virtual environment and install the packages.

```bash
conda create -n neuralemb python=3.9
conda activate neuralemb
conda install -c conda-forge mamba -y
mamba install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge -y
mamba install -y -c bioconda -c conda-forge snakemake -y
mamba install -c conda-forge graph-tool scikit-learn numpy numba scipy pandas networkx seaborn matplotlib gensim ipykernel tqdm black faiss=1.7.3 -y
```

Install the in-house packages 

```bash
cd libs/BeliefPropagation && pip install -e . 
cd libs/embcom && pip install -e . 
```

Install the Python wrapper for the LFR network generator:

```bash 
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rUMowBj13WDDsZ_s6td-Fxw1qsLNIn6Z' -O - --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rUMowBj13WDDsZ_s6td-Fxw1qsLNIn6Z' -qO- | tar -xz
mv binary_networks libs/lfr_benchmark/lfr_benchmark/lfr-generator
cd libs/lfr_benchmark/lfr_benchmark/lfr-generator && make
```

Then, create file `config.yaml` with the following content:
``yaml
data_dir: "data/"
```
and place the yaml file under `workflow` folder. Note that the script will generate over 1T byte of data under this data folder. So make sure you have sufficient space.


## Run simulation  

Run the `Snakemake`:

```bash 
snakemake --cores 24 all 
```


# About the code

## Graph embedding

We provide a ready-to-use package for graph embedding methods, including node2vec, DeepWalk, LINE, and some conventional graph embedding.
The package can be install by 

```bash 
cd libs/embcom && pip install -e . 
```

### Usage

```python 
import embcom
import networkx as nx 

# Load the network for demonstration
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)

# Loading the node2vec model
# `window_length`: Size of window, T
# `num_walks`: Number of walks
model = embcom.embeddings.Node2Vec(window_length=80, num_walks=20)

# Train
# `net`: scipy sparse matrix
model.fit(net) 

# Generate an embedding
# `dim`: Integer 
model.transform(dim=dim) 
```

Other embedding models:
````python
# DeepWalk 
model = embcom.embeddings.DeepWalk(window_length=window_length, num_walks=num_walks)

# Laplacian EigenMap
model = embcom.embeddings.LaplacianEigenMap()

# Laplacian EigenMap
model = embcom.embeddings.LaplacianEigenMap()

# Modularity spectral Embedding
model = embcom.embeddings.ModularitySpectralEmbedding()

# Non-backtracking spectral embedding 
model = embcom.embeddings.NonBacktrackingSpectralEmbedding()
```


## Belief propagation algorithm 

We develop a Python wrapper for the belief propagation method. See (this package)[https://github.com/skojaku/BeliefPropagation] for the details.
