# Reproducing the Results

## Setup

1. Create and activate the conda environment:
```bash
mamba env create -f environment.yml
conda activate neuralemb
```

2. Install the in-house packages:
```bash
pip install -e libs/embcom
pip install -e libs/BeliefPropagation
pip install -e libs/LFR-benchmark
```

4. Create `workflow/config.yaml`:
```yaml
data_dir: "data/"
```

Note: the pipeline generates over 1 TB of data under `data/`. Make sure you have sufficient disk space.

## Running the Pipeline

```bash
snakemake --cores 24 all   # full pipeline (networks + embeddings + evaluation + figures)
snakemake --cores 24 figs  # figures only (after data exists)
```

Partial runs targeting a single dataset:
```bash
snakemake --cores 24 all_mpm  # multi-partition model only
snakemake --cores 24 all_lfr  # LFR benchmark only
snakemake --cores 24 all_emp  # empirical networks only
```

Adjust `--cores` to match your machine.

## Code Organization

### Repository Structure

```
.
├── Snakefile                        # Main pipeline entry point
├── workflow/
│   ├── config.yaml                  # Pipeline configuration (data_dir)
│   ├── rules/                       # Snakemake rule modules
│   │   ├── common.smk               # Shared utility functions
│   │   ├── multipartition.smk       # Rules for multi-partition model experiments
│   │   ├── lfr.smk                  # Rules for LFR benchmark experiments
│   │   └── empirical.smk            # Rules for empirical network experiments
│   └── scripts/                     # Python scripts called by Snakemake rules
│       ├── generate_net_by_multi_partition_model.py
│       ├── generate_lfr_net.py
│       ├── generate_empirical_network.py
│       ├── embedding.py
│       ├── detect_community.py
│       ├── kmeans_clustering.py
│       ├── voronoi_clustering.py
│       ├── silhouette_kmeans.py
│       ├── eval_com_detect_score.py
│       ├── concatenate_results.py
│       ├── plot_mixing_vs_performance.py
│       ├── plot_mixing_vs_performance_lfr.py
│       ├── plot_performance_empirical_net.py
│       ├── pdf_nup.py
│       └── color_palette.py
├── libs/
│   ├── embcom/                      # Graph embedding library
│   ├── BeliefPropagation/           # Belief propagation C wrapper
│   └── LFR-benchmark/               # LFR benchmark C wrapper
└── data/                            # Generated data (created by the pipeline)
    └── {dataset}/
        ├── networks/                # Sparse adjacency matrices (.npz)
        ├── embedding/               # Embedding vectors (.npz)
        ├── communities/             # Detected community labels (.npz)
        └── evaluations/             # Scores and all-result.csv
```

### Snakemake Rule Files

| File | Purpose |
|---|---|
| `Snakefile` | Entry point — parameter definitions, top-level rules (`all`, `figs`, `all_mpm`, `all_lfr`, `all_emp`) |
| `workflow/rules/common.smk` | Helper functions (`make_filename`, `to_paramspace`) |
| `workflow/rules/multipartition.smk` | All rules for the planted partition model dataset |
| `workflow/rules/lfr.smk` | All rules for the LFR benchmark dataset |
| `workflow/rules/empirical.smk` | All rules for empirical networks |

### Pipeline Stages

The pipeline is a five-stage DAG managed by Snakemake:

```
Network Generation → Embedding → Community Detection / Clustering → Evaluation → Figures
```

#### 1. Network Generation

Three datasets are supported:

| Dataset | Rule file | Script | Description |
|---|---|---|---|
| `multi_partition_model` | `rules/multipartition.smk` | `generate_net_by_multi_partition_model.py` | Synthetic networks via planted partition model (igraph) |
| `lfr` | `rules/lfr.smk` | `generate_lfr_net.py` | Synthetic networks via LFR benchmark |
| `empirical` | `rules/empirical.smk` | `generate_empirical_network.py` | Real-world networks (polblog, airport, football, polbooks, cora, highschool) |

Output: scipy sparse matrices saved as `.npz` in `data/{dataset}/networks/`.

#### 2. Embedding

`embedding.py` applies one of seven graph embedding models from the `embcom` library:

| Key | Model |
|---|---|
| `node2vec` | Node2Vec |
| `deepwalk` | DeepWalk |
| `line` | LINE |
| `leigenmap` | Laplacian EigenMap |
| `adjspec` | Adjacency Spectral Embedding |
| `modspec` | Modularity Spectral Embedding |
| `nonbacktracking` | Non-Backtracking Spectral Embedding |

Output: embedding vectors saved as `.npz` in `data/{dataset}/embedding/`.

#### 3. Community Detection

Two paths run in parallel:

**Direct detection** (`detect_community.py`): Applies Infomap, flat SBM (graph-tool), or belief propagation directly to the network.

**Embedding-based clustering**: Assigns communities by clustering the embedding vectors using one of three methods:

| Method | Script |
|---|---|
| K-means | `kmeans_clustering.py` |
| Voronoi | `voronoi_clustering.py` |
| Silhouette-based K-means | `silhouette_kmeans.py` |

Output: community label arrays saved as `.npz` in `data/{dataset}/communities/`.

#### 4. Evaluation

`eval_com_detect_score.py` computes NMI and element-centric similarity (esim) against ground-truth communities. `concatenate_results.py` then merges all individual score files into `data/{dataset}/evaluations/all-result.csv`.

#### 5. Figures

Three plotting scripts read `all-result.csv` and produce PDF figures under `figs/`:

| Script | Figure |
|---|---|
| `plot_mixing_vs_performance.py` | Performance vs. mixing parameter (multi-partition model) |
| `plot_mixing_vs_performance_lfr.py` | Performance vs. mixing parameter (LFR benchmark) |
| `plot_performance_empirical_net.py` | Performance on empirical networks |

`pdf_nup.py` tiles individual figure PDFs into a grid layout for the paper.

### Parameter Space

The full parameter grid is defined at the top of `Snakefile`. Key parameters:

| Parameter | Production values | Test values (default) |
|---|---|---|
| Network size `n` | 10,000 / 100,000 | 500 |
| Embedding dimension `dim` | 16, 64, 128 | 16 |
| Mixing parameter `mu` | 19 values in [0.1, 1] | 5 values |
| Average degree `cave` | 5, 10, 50 | 5 |

Test values (smaller and faster) are the defaults in `Snakefile`; production values are commented out alongside them.
