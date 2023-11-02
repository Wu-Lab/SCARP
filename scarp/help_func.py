import numpy as np
import scanpy as sc


def getNClusters_Louvain(adata, n_cluster, range_min=0, range_max=3, max_steps=50, random_seed=1):
    np.random.seed(random_seed)
    temp_step = 0
    temp_min = float(range_min)
    temp_max = float(range_max)
    while temp_step < max_steps:
        temp_resolution = temp_min + ((temp_max - temp_min) / 2)
        sc.tl.louvain(adata, resolution=temp_resolution)
        temp_clusters = adata.obs['louvain'].nunique()

        if temp_clusters > n_cluster:
            temp_max = temp_resolution
        elif temp_clusters < n_cluster:
            temp_min = temp_resolution
        else:
            return [1, adata]
        temp_step += 1

    print('Cannot find the number of clusters')
    print('Clustering solution from last iteration is used:' +
          str(temp_clusters) + ' at resolution ' + str(temp_resolution))
    sc.tl.louvain(adata, resolution=temp_resolution)
    adata.obs['louvain'].nunique()
    return [0, adata]


def getNClusters_Leiden(adata, n_cluster, range_min=0, range_max=3, max_steps=50, random_seed=1):
    np.random.seed(random_seed)
    temp_step = 0
    temp_min = float(range_min)
    temp_max = float(range_max)
    while temp_step < max_steps:
        temp_resolution = temp_min + ((temp_max - temp_min) / 2)
        sc.tl.leiden(adata, resolution=temp_resolution)
        temp_clusters = adata.obs['leiden'].nunique()

        if temp_clusters > n_cluster:
            temp_max = temp_resolution
        elif temp_clusters < n_cluster:
            temp_min = temp_resolution
        else:
            return [1, adata]
        temp_step += 1

    print('Cannot find the number of clusters')
    print('Clustering solution from last iteration is used:' +
          str(temp_clusters) + ' at resolution ' + str(temp_resolution))
    sc.tl.leiden(adata, resolution=temp_resolution)
    adata.obs['leiden'].nunique()
    return [0, adata]