import os

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy, vq
import matplotlib.pyplot as plt

if __name__ == "__main__":
    for path, subdirs, files in os.walk('./HMP_Dataset'):
        if path == './HMP_Dataset' or 'MODEL' in path or 'Brush' not in path:
            continue
        for file in files:
            print(os.path.join(path, file))
            all_data = pd.read_csv(os.path.join(path, file),
                                   names=None,
                                   na_values=['?'], delimiter=r"\s+")
            no_of_split = int(all_data.iloc[:, 0].shape[0] / 32)
            remainder = all_data.iloc[:, 0].shape[0] % 32
            clean_split_data = all_data.iloc[:all_data.shape[0] - remainder]
            split_all_data = np.vsplit(clean_split_data,no_of_split)
            split_all_data.append(all_data.iloc[-32:])



            for i in range(0,len(split_all_data)):
                data = pd.concat([split_all_data[i].iloc[:, 0].T, split_all_data[i].iloc[:, 1].T, split_all_data[i].iloc[:, 2].T], axis=0)
            print('d')
