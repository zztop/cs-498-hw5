import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

if __name__ == "__main__":
    all_data = pd.read_csv('./jobs.csv',
                           names=None,
                           na_values=['?'], sep='\t')
    all_data = all_data.dropna()
    all_data_countries = all_data["Country"]
    all_data = all_data.drop(all_data.columns[[0]], axis=1)

    error = []
    error_ec = {}
    K = range(1, all_data.shape[1])
    for k in K:
        kmeans = KMeans(n_clusters=k)
        means2 = kmeans.fit_predict(all_data.values)
        mean_cluster_centers = np.mean(kmeans.cluster_centers_, axis=0)
        # error[k] = np.sum((all_data.values - mean_cluster_centers) ** 2, axis=1)
        error.append(sum(np.min(cdist(all_data.values, kmeans.cluster_centers_, 'euclidean'), axis=1)))

    plt.figure()
    plt.plot(K, error, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.savefig('elbow.png')
    plt.close()

    plt.figure()
    single_linkage = hierarchy.single(all_data)
    dn = hierarchy.dendrogram(single_linkage)
    plt.savefig('single.png')
    plt.close()

    plt.figure()
    complete_linkage = hierarchy.complete(all_data)
    dn = hierarchy.dendrogram(complete_linkage)
    plt.savefig('complete_linkage.png')
    plt.close()

    plt.figure()
    average_linkage = hierarchy.average(all_data)
    dn = hierarchy.dendrogram(average_linkage)
    plt.savefig('average_linkage.png')
    plt.close()

    print('done')
