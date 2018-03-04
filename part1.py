import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy, vq
import matplotlib.pyplot as plt

if __name__ == "__main__":
    all_data = pd.read_csv('./jobs.csv',
                           names=None,
                           na_values=['?'], sep='\t')
    all_data = all_data.dropna()
    all_data_countries = all_data["Country"]
    all_data = all_data.drop(all_data.columns[[0]], axis=1)

    kmeans = KMeans(n_clusters=3)
    means1 = kmeans.fit_predict(all_data.values)

    error = []
    error_ec = {}
    K=range(1, all_data.shape[1])
    for k in K:
        kmeans = KMeans(n_clusters=k)
        means2 = kmeans.fit_predict(all_data.values)
        mean_cluster_centers = np.mean(kmeans.cluster_centers_,axis=0)
        # error[k] = np.sum((all_data.values - mean_cluster_centers) ** 2, axis=1)
        error.append(sum(np.min(cdist(all_data.values,kmeans.cluster_centers_, 'euclidean'),axis=1)))

    plt.figure()
    plt.plot(K,error,'bx-')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.savefig('elbow.png')


    kmeans = KMeans(n_clusters=5)
    mean3 = kmeans.fit_predict(all_data)

    kmeans = KMeans(n_clusters=6)
    mean4 = kmeans.fit_predict(all_data)

    plt.figure()
    single_linkage = hierarchy.single(all_data)
    dn = hierarchy.dendrogram(single_linkage)
    plt.show()

    plt.figure()
    complete_linkage = hierarchy.complete(all_data)
    dn = hierarchy.dendrogram(complete_linkage)
    plt.show()

    plt.figure()
    average_linkage = hierarchy.average(all_data)
    dn = hierarchy.dendrogram(average_linkage)
    plt.show()

    print('done')
