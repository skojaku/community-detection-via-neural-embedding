# Project template

A simple template for research project repos. Also check out [data science and
reproducible science cookie
cutters](https://github.com/audreyr/cookiecutter#data-science).

## Installation

Run the following

    ./install.sh YOUR_PROJECT_REPO_FOLDER

This script creates the following folders and files. 

1. `libs` for a software library for the project.
1. `data` for datasets and scripts for downloading datasets.
1. `exps` for timestamped experiments.
1. `paper` for manuscripts.
1. `workflow` for workflow scripts.
1. `.gitignore` that lists temporary and binary files to ignore (LaTeX, Python, Jupyter, data files, etc. )

## Set up

### Anaconda

First create a virtual environment for the project.

    conda create -n project_env_name python=3.7
    conda activate project_env_name

Install `ipykernel` for Jupyter and `snakemake` for workflow management. 

    conda install -y -c bioconda -c conda-forge snakemake ipykernel numpy pandas scipy matplotlib seaborn tqdm austin

Create a kernel for the virtual environment that you can use in Jupyter lab/notebook.

    python -m ipykernel install --user --name project_env_kernel_name


### Git pre-commit

```bash
conda install -y -c conda-forge pre-commit
pre-commit install
```


### Snakemake setting 

```bash 
mkdir -p ~/.config/snakemake/default 
```
and create `~/.config/snakemake/default/config.yaml`:
```yaml
# non-slurm profile defaults
keep-going: True
rerun-triggers: mtime
```

and add the following to .zshrc or .bashrc file
```bash 
export SNAKEMAKE_PROFILE=default
```
