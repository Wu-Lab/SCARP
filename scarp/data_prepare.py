import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix
import re



'''
#########################################################################################################
#                 various data format transformation & data pre-processing functions
#########################################################################################################
'''


# construct adata from standard 10X data: three files should be in 'root_file':
# barcodes.tsv     peaks.tsv       counts.mtx
def construct_AnnData_10X(root_file, min_cells, data_name, save=True, save_path=None):
    barcodes = pd.read_table(root_file + 'barcodes.tsv', header=None)
    peaks = pd.read_table(root_file + 'peaks.tsv', header=None)
    matrix = sc.read(root_file + 'counts.mtx', header=None)

    adata = sc.AnnData(matrix.X.T)
    adata.var = pd.DataFrame(index=peaks[0].to_list())
    adata.obs = pd.DataFrame(index=barcodes[0].to_list())

    sc.pp.filter_genes(adata, min_cells=min_cells)

    if save:
        adata.write(save_path + data_name + '.h5ad')

    return adata


# barcodes: list type
# peaks: list type
# matrix: csc or csr type,
def construct_AnnData(barcodes, peaks, matrix, min_cells, min_counts,
                      data_name, save=True, save_path=None):
    adata = sc.AnnData(matrix)
    adata.var = pd.DataFrame(index=peaks)
    adata.obs = pd.DataFrame(index=barcodes)

    adata = sort_peaks(adata)
    sc.pp.filter_genes(adata, min_cells=min_cells, min_counts=min_counts)

    if save:
        adata.write(save_path + data_name + '.h5ad')

    return adata


# change a matrix (cell*peak) to DataFrame with 3 columns: Peaks, Cells, Counts
def construct_mat_to_3col(mat):
    Cells_num, Peaks_num = mat.shape
    data_unpivot = {
        "Counts": mat.to_numpy().ravel("F"),
        "Peaks": np.asarray(mat.columns).repeat(Cells_num),
        "Cells": np.tile(np.asarray(mat.index), Peaks_num),
    }
    return pd.DataFrame(data_unpivot, columns=["Peaks", "Cells", "Counts"])


# construct sparse matrix of shape (N,N) or (Cells_num, Peaks_num)
# from Count_df with 3 columns: Peaks (start from 1); Cells (start from 1); Counts;
def construct_3col_to_sparse_mat(Count_df, normalization=True, binarization=True, return_shape='NN'):
    Peaks_num = Count_df['Peaks'].max()
    Cells_num = Count_df['Cells'].max()
    N = Peaks_num + Cells_num

    if binarization:
        sparse_mat = csc_matrix((np.ones(Count_df.shape[0], dtype='int'),
                                 (Count_df['Cells'] - 1, Count_df['Peaks'] - 1 + Cells_num)),
                                shape=(N, N))
    else:
        sparse_mat = csr_matrix((Count_df['Counts'],
                                 (Count_df['Cells'] - 1, Count_df['Peaks'] - 1 + Cells_num)),
                                shape=(N, N))

    sparse_mat = sparse_mat + sparse_mat.T
    if normalization:
        sparse_mat = Net_normalization(sparse_mat)

    if return_shape == 'NN':
        return sparse_mat
    elif return_shape == 'CP':
        return sparse_mat[0:Cells_num, Cells_num:]
    else:
        print('Wrong shape! Should be NN or CP.')


# input matrix type: csc or csr
# input matrix shape: (N, N)
def Net_normalization(sparse_mat):
    n = sparse_mat.shape[0]
    Norm_factor = csr_matrix((np.array(np.sqrt(1 / sparse_mat.sum(1))).ravel().tolist(),
                              (range(n), range(n))), shape=(n, n))

    sparse_mat = Norm_factor * sparse_mat * Norm_factor
    return sparse_mat


# input matrix type: csc or csr
# input matrix shape: (Cells_num, Peaks_num)
def Mat_normalization(sparse_mat):
    m, n = sparse_mat.shape
    Norm_factor1 = csr_matrix((np.array(np.sqrt(1 / sparse_mat.sum(1))).ravel().tolist(),
                               (range(m), range(m))), shape=(m, m))
    Norm_factor2 = csr_matrix((np.array(np.sqrt(1 / sparse_mat.sum(0))).ravel().tolist(),
                               (range(n), range(n))), shape=(n, n))

    sparse_mat = Norm_factor1 * sparse_mat * Norm_factor2

    return sparse_mat


def sort_peaks(adata):
    Peaks = adata.var.index.tolist()

    # delete peaks that are not indexed by 'Chr'
    judge = ['chr' in Peaks[i] for i in range(len(Peaks))]
    chr_keep = np.where(judge)[0]
    Peaks = [Peaks[i] for i in chr_keep]

    # delete M chromosome
    judge = ['chrM' not in Peaks[i] for i in range(len(Peaks))]
    chr_keep = np.where(judge)[0]
    Peaks = [Peaks[i] for i in chr_keep]

    Peaks_array = np.array([re.split('[\W+_]', i) for i in Peaks])
    chri2i = dict(zip(['chr' + str(i) for i in range(1, 23)], range(1, 23)))
    chri2i['chrX'] = 23
    chri2i['chrY'] = 24
    Peaks_array[:, 0] = [chri2i[i] for i in Peaks_array[:, 0]]
    Peaks_df = pd.DataFrame(Peaks_array.astype('float'), columns=['chr', 'from', 'to'])
    Peaks_df = Peaks_df.sort_values(by=['chr', 'from'])

    # cell*peaks matrix, reordering of columns
    X = adata.X.todense()
    X = X[:, np.where(judge)[0]]
    sparse_df = pd.DataFrame(X)
    sparse_df[list(range(sparse_df.shape[1]))] = sparse_df[Peaks_df.index]

    adata_new = sc.AnnData(csr_matrix(sparse_df))
    adata_new.var = pd.DataFrame(index=[Peaks[i] for i in Peaks_df.index])
    adata_new.obs = adata.obs

    return adata_new


def filter_cells(adata, keep_cells):
    adata_df = pd.DataFrame(adata.X.todense(),
                            index=adata.obs.index,
                            columns=adata.var.index)
    adata_new = sc.AnnData(adata_df.loc[keep_cells])
    adata_new.obs = adata.obs.loc[keep_cells]
    adata_new.var = adata.var
    adata_new.X = csr_matrix(adata_new.X)
    return adata_new


def filter_peaks(adata, keep_peaks):
    adata_df = pd.DataFrame(adata.X.todense(),
                            index=adata.obs.index,
                            columns=adata.var.index)
    adata_new = sc.AnnData(adata_df[keep_peaks])
    adata_new.obs = adata.obs
    adata_new.var = adata.var.loc[keep_peaks]
    adata_new.X = csr_matrix(adata_new.X)
    return adata_new