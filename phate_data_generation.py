import pandas
import matplotlib.pyplot as plt
import os
import numpy as np
import phate
import scipy.stats

d=100
n=10

tree_data, tree_clusters = phate.tree.gen_dla(n_dim=d, n_branch=n,  branch_length=1000)
norm = (tree_data - tree_data.min())
norm = norm/norm.max()

pandas.DataFrame(norm).to_csv("./data/X.csv",header=None,index=False)
pandas.DataFrame(tree_clusters).to_csv("./data/labels.csv",header=None,index=False)
