import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Visualize the embedding obtained after running an experiment.')
parser.add_argument('--path', help='Path to the experiment (e.g. experiments/template-finetuning)')
args = parser.parse_args()

print(f"Plotting visualization for experiment : {args.path}")
print("Loading embedding ...")
df_ft = pd.read_csv(args.path + "/embedding.csv",header=None).to_numpy()

print("Plotting ...")
plt.figure(figsize=(8,5))
plt.scatter(*df_ft.T,s=5,alpha=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.show()

