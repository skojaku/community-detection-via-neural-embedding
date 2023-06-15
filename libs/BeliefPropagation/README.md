# BeliefPropagation
Belief propagation method for community detection. 

This is a Python wrapper of C++ code by Aurelien Decelle, Florent Krzakala, Lenka Zdeborova and Pan Zhang. 
The original code can be obtained from [here](http://home.itp.ac.cn/~panzhang/).


# Install

```bash
git clone https://github.com/skojaku/BeliefPropagation
cd BeliefPropagation
python setup.py build
pip install -e .
```

You also need the following packages:
- `pandas`
- `networkx`
- `scipy`
- `numpy`

# Usage

```python 
import belief_propagation
import networkx as nx

# Create a karate club network.
A = nx.to_scipy_sparse_matrix(nx.karate_club_graph()) 

# Detect communities
# A: scipy sparse matrix. CSR format. 
# q: Number of communities
community_ids = belief_propagation.detect(A, q=3) 
```






