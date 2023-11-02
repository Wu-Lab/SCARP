import time
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from .data_prepare import Mat_normalization


def SCARP(adata=None,
          data_name=None,
          m=1.5,
          gamma=3000,
          beta=5000,
          return_shape='CN',
          peak_loc=True,
          parallel=0,
          plot_SD=True,
          fig_size=(4, 3),
          save_file=None,
          verbose=True
          ):
    """
    :param adata: h5ad format, input scATAC-seq data
    :param data_name: name of this dataset
    :param m: parameter to control NR diffusion intensity
    :param gamma: parameter to control the threshold for merging adjacent chromosomes
    :param beta: parameter to control the extent to which prior edge weight decays
    :param return_shape: shape of the returned matrix
    :param peak_loc: use peak location prior information or not
    :param parallel: parallel computing or not. 0 means automatically determined
    :param plot_SD: plot the SDs of PCs or not
    :param fig_size: figure size of the SD plot
    :param save_file: if plot_std is True, the file path you want to save
    :param verbose: print the process or not
    :return: Cell embeddings dataframe
    """
    Cells_num, Peaks_num = adata.X.shape
    Cells = adata.obs.index

    t, diffusion_mat = SCARP_diffusion_mat(adata=adata,
                                           m=m,
                                           gamma=gamma,
                                           beta=beta,
                                           return_shape=return_shape,
                                           peak_loc=peak_loc,
                                           parallel=parallel,
                                           verbose=verbose
                                           )
    k = SCARP_SD_plot(data=diffusion_mat,
                      peaks_num=Peaks_num,
                      title=data_name,
                      plot_SD=plot_SD,
                      fig_size=fig_size,
                      save_file=save_file
                      )

    Cells_df = SCARP_cell_embedding(diffusion_mat=diffusion_mat,
                                    kept_comp=k,
                                    Cells=Cells)

    return Cells_df


def SCARP_diffusion_mat(adata,
                        m=1.5,
                        gamma=3000,
                        beta=5000,
                        return_shape='CN',
                        peak_loc=True,
                        parallel=0,
                        verbose=True
                        ):
    """
    :param adata: h5ad format, input scATAC-seq data
    :param m: parameter to control NR diffusion intensity
    :param gamma: parameter to control the threshold for merging adjacent chromosomes
    :param beta: parameter to control the extent to which prior edge weight decays
    :param return_shape: shape of the returned matrix
    :param peak_loc: use peak location prior information or not
    :param parallel: parallel computing or not. 0 means automatically determined
    :param verbose: print the process or not
    :return: NR diffused matrix
    """
    Cells_num, Peaks_num = adata.X.shape
    sparse_matrix = adata.X  # sparse matrix
    sparse_matrix = (sparse_matrix > 0) * 1  # binary
    sparse_matrix = Mat_normalization(sparse_matrix)  # normalization

    # ==================================================================
    # Peaks information
    Peaks = adata.var.index.tolist()
    Peaks_array = np.array([re.split('_|\W+', i) for i in Peaks])
    chri2i = dict(zip(['chr' + str(i) for i in range(1, 23)], range(1, 23)))
    chri2i['chrX'] = 23
    chri2i['chrY'] = 24
    Peaks_array[:, 0] = [chri2i[i] for i in Peaks_array[:, 0]]
    Peaks_df = pd.DataFrame(Peaks_array.astype(
        'float'), columns=['chr', 'from', 'to'])
    Peaks_df = Peaks_df.sort_values(by=['chr', 'from'])

    # merge small chromosome
    Peaks_df = merge_small_chromosome(Peaks_df, gamma)
    Peaks_df['merge_chr'] = Peaks_df['merge_chr'].astype('str')

    # ==================================================================
    '''
    Divide and conquer.
    Divide the whole genome into 24 sets of chromosomes,
    and construct a cell*peak bipartite map inside each chromosome,
    where peak includes only the genomic information of the current chromosome
    '''
    groups = Peaks_df.groupby('merge_chr')
    chrs = Peaks_df['merge_chr'].unique()

    diffusion_dict = {}
    peak_to = [0]
    for i in chrs:
        peak_to.append(peak_to[-1] + groups.get_group(i).shape[0])
    peak_to = peak_to[0:-1]

    # auto
    if parallel == 0:
        if (Peaks_num < 50000) or (Cells_num > 5000):
            if verbose:
                print(
                    '\n%%%%%%%%%%%%%%%%%%%%%%%% Diffusion Started (without parallel computing)%%%%%%%%%%%%%%%%%%%%%%%%%%')
            t1 = time.time()
            for i in range(chrs.shape[0]):
                diffusion_dict[chrs[i]] = single_chromosome_diffusion(groups, chrs[i], peak_to[i],
                                                                      sparse_matrix, Cells_num, beta, m, peak_loc,
                                                                      verbose)
        else:
            num_cores = min(multiprocessing.cpu_count(), 23)

            if verbose:
                print('\n%%%%%%%%%%%%%%%%%%%%%%% Diffusion Started (with ' + str(
                    num_cores) + ' cores parallel computing)%%%%%%%%%%%%%%%%%%%%%%%')
            t1 = time.time()
            results = Parallel(n_jobs=num_cores, temp_folder='./')(delayed(single_chromosome_diffusion)(
                groups, chrs[i], peak_to[i], sparse_matrix, Cells_num, beta, m, peak_loc, verbose) for i in
                                                                   range(chrs.shape[0]))
            for i in range(chrs.shape[0]):
                diffusion_dict[chrs[i]] = results[i]

    elif parallel == 1:
        if verbose:
            print('\n%%%%%%%%%%%%%%%%%%%%%%%% Diffusion Started (without parallel computing)%%%%%%%%%%%%%%%%%%%%%%%%%%')
        t1 = time.time()
        for i in range(chrs.shape[0]):
            diffusion_dict[chrs[i]] = single_chromosome_diffusion(groups, chrs[i], peak_to[i],
                                                                  sparse_matrix, Cells_num, beta, m, peak_loc, verbose)
    else:
        parallel = min(parallel, 24)
        if verbose:
            print('\n%%%%%%%%%%%%%%%%%%%%%%% Diffusion Started (with ' + str(parallel) + 'cores parallel '
                                                                                         'computing)%%%%%%%%%%%%%%%%%%%%%%%')
        t1 = time.time()
        results = Parallel(n_jobs=parallel, temp_folder='./')(delayed(single_chromosome_diffusion)(
            groups, chrs[i], peak_to[i], sparse_matrix, Cells_num, beta, m, peak_loc, verbose) for i in
                                                              range(chrs.shape[0]))
        for i in range(chrs.shape[0]):
            diffusion_dict[chrs[i]] = results[i]

    if verbose:
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%% Diffusion Finished %%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    t2 = time.time()
    t_diffusion = t2 - t1
    if verbose:
        print('Running time: ' + str(t_diffusion) + 's')

    # ==================================================================
    if verbose:
        print('\nMatrix Splicing...........')
    t1 = time.time()

    if return_shape == 'NN':
        # =====CC=====
        diffusion_cellmat_ave = np.zeros((Cells_num, Cells_num))
        for i in chrs:
            diffusion_cellmat_ave = diffusion_cellmat_ave + \
                                    diffusion_dict[i][0:Cells_num, 0:Cells_num]
        # =====CP, PC=====
        CP = sp.hstack([diffusion_dict[i][0:Cells_num, Cells_num:]
                        for i in chrs])

        diffusion_mat = sp.vstack([sp.hstack([diffusion_cellmat_ave / len(chrs), CP]),
                                   sp.hstack(
                                       [CP.T, sp.csc_matrix((Peaks_num, Peaks_num))])
                                   ])

        diffusion_mat = sp.lil_matrix(diffusion_mat)
        # =====PP=====
        peak_to = Cells_num
        for i in chrs:
            diffusion_mat_i = diffusion_dict[i]
            peak_num_i = diffusion_mat_i.shape[0] - Cells_num
            peak_from = peak_to
            peak_to = peak_from + peak_num_i
            diffusion_mat[peak_from:peak_to,
            peak_from:peak_to] = diffusion_mat_i[Cells_num:, Cells_num:]  # PP
        t2 = time.time()
        t_splic = t2 - t1
        if verbose:
            print('Splicing time: ' + str(t_splic) + 's')
        diffusion_mat = sp.csr_matrix(diffusion_mat)

    elif return_shape == 'CN':
        # =====CC=====
        diffusion_cellmat_ave = np.zeros((Cells_num, Cells_num))
        for i in chrs:
            diffusion_cellmat_ave = diffusion_cellmat_ave + \
                                    diffusion_dict[i][0:Cells_num, 0:Cells_num]
        # =====CP=====
        CP = sp.hstack([diffusion_dict[i][0:Cells_num, Cells_num:]
                        for i in chrs])

        diffusion_mat = sp.hstack([diffusion_cellmat_ave / len(chrs), CP])
        t2 = time.time()
        t_splic = t2 - t1
        if verbose:
            print('Splicing time: ' + str(t_splic) + 's')
        diffusion_mat = sp.csr_matrix(diffusion_mat)

    elif return_shape == 'CP':
        # =====CP=====
        diffusion_mat = sp.hstack(
            [diffusion_dict[i][0:Cells_num, Cells_num:] for i in chrs])
        t2 = time.time()
        t_splic = t2 - t1
        if verbose:
            print('Splicing time: ' + str(t_splic) + 's')
        diffusion_mat = sp.csr_matrix(diffusion_mat)

    else:
        print('Wrong return shape, should be NN, CN, or CP')
        t_splic = None
        diffusion_mat = None

    return str(t_diffusion + t_splic), diffusion_mat


# input matrix type: numpy array
# return matrix type: csr
def NR_F(mat, m=1.5):
    # -------------------------------------------------------------------
    # change into transition matrix
    if type(mat) == sp.csc_matrix or type(mat) == sp.csr_matrix:
        mat = np.array(mat.todense())
    mat = np.array(mat)  # it's fast to compute using numpy array format
    N = mat.shape[0]
    P1 = mat / mat.sum(1).reshape(-1, 1)

    # diffusion
    P2 = (m - 1) * P1.dot(np.linalg.inv(m * np.eye(N) - P1))
    output_network = np.diag(mat.sum(1)).dot(P2)

    return sp.csr_matrix(output_network)


def single_chromosome_diffusion(groups, i, peak_to, sparse_matrix, Cells_num, beta, m, peak_loc=True, verbose=True):
    groups_i = groups.get_group(i)  # Current chromosome peak information
    peak_num_i = groups_i.shape[0]
    if '_' not in i:
        if verbose:
            print('       ====================== Chr' + str(i) + ': ' + str(
                peak_num_i) + ' Peaks ========================')
    else:
        info_temp = [int(i) for i in re.split('_|\W+', i)]
        if verbose:
            print('       ====================== Chr' + str(min(info_temp)) + '~' + str(max(info_temp)) +
                  ': ' + str(peak_num_i) + ' Peaks ========================')

    peak_from = peak_to
    peak_to = peak_from + peak_num_i
    #  =======Bipartite graph of the current chromosome===========
    sparse_i = sparse_matrix[:, peak_from:peak_to]
    sparse_i = np.array(sparse_i.todense())
    sparse_mat_i = np.vstack([np.hstack([np.zeros((Cells_num, Cells_num)), sparse_i]),
                              np.hstack([sparse_i.T, np.zeros((peak_num_i, peak_num_i))])])

    #  =======The isolated points of bipartite graph are removed first, and then added back later===========
    kept_index = np.where(sparse_mat_i.sum(1) == 0)[0]
    if kept_index.shape[0] != 0:
        sparse_mat_i = np.delete(sparse_mat_i, kept_index, axis=0)
        sparse_mat_i = np.delete(sparse_mat_i, kept_index, axis=1)
        # ==============================================================
        if peak_loc:
            # # Padding peaks information
            peak_max = np.max(sparse_mat_i)
            Padding_mat = sparse_mat_i.copy()
            padding = create_peak_loc_subdiag_mat(
                groups_i, peak_max, beta)  # peaks*peaks matrix
            Padding_mat[(Cells_num - kept_index.shape[0]):,
            (Cells_num - kept_index.shape[0]):] = padding
        else:
            Padding_mat = sparse_mat_i
        # ==============================================================
        diffusion_temp = NR_F(Padding_mat, m)
        diffusion_temp = np.array(diffusion_temp.todense())
        for kk in range(kept_index.shape[0]):
            diffusion_temp = np.insert(diffusion_temp, kept_index[kk],
                                       np.zeros(diffusion_temp.shape[0]), axis=1)
            diffusion_temp = np.insert(diffusion_temp, kept_index[kk],
                                       np.zeros(diffusion_temp.shape[1]), axis=0)
        diffusion_ = sp.csr_matrix(diffusion_temp)
    else:
        # ==============================================================
        if peak_loc:
            # Padding peaks information
            peak_max = np.max(sparse_mat_i)
            Padding_mat = sparse_mat_i.copy()
            Padding_mat[Cells_num:, Cells_num:] = create_peak_loc_subdiag_mat(groups_i, peak_max,
                                                                              beta)  # peaks*peaks matrix
        else:
            Padding_mat = sparse_mat_i
        # ==============================================================
        diffusion_ = NR_F(Padding_mat, m)

    return diffusion_


# Max_value: Maximum value of peaks*peaks matrix padding
# beta: the parameter for gene mapping function
def create_peak_loc_subdiag_mat(Peaks_df, Max_value, beta):
    chrs_temp = Peaks_df['chr'].unique()

    # break peak between chromosomes
    break_peak = np.zeros(chrs_temp.shape[0] - 1).astype(int)
    for i in range(chrs_temp.shape[0] - 1):
        chr_i = chrs_temp[i]
        break_peak[i] = max(np.where(Peaks_df['chr'] == chr_i)[0])

    Peaks_df = np.array(Peaks_df)
    # compute the number of bases (subdiagonal)
    Peaks_df_subdiag = Peaks_df[1:, 1] - Peaks_df[0:-1, 2]
    Peaks_df_subdiag[break_peak] = 0
    Peaks_df_subdiag[Peaks_df_subdiag < 0] = 0
    Peaks_df_subdiag = Peaks_df_subdiag.astype('float')

    # change into matrix weight
    peaks_dist = np.exp((-2) * Peaks_df_subdiag / beta)
    peaks_dist = Max_value * peaks_dist / peaks_dist.max()
    peaks_dist_mat = np.diag(peaks_dist, k=1)
    peaks_dist_mat = peaks_dist_mat + peaks_dist_mat.T

    return peaks_dist_mat


def merge_small_chromosome(Peaks_df, merge_thre):
    Peaks_df['chr'] = Peaks_df['chr'].astype(int)
    Peaks_num_stat = Peaks_df['chr'].value_counts().sort_index()
    Peaks_num_stat.index = Peaks_num_stat.index.astype('str')
    Peaks_num_stat = merge_(Peaks_num_stat, merge_thre)
    Peaks_df['merge_chr'] = Peaks_df['chr'].astype(str)
    for i in Peaks_num_stat.index:
        mer = re.split('_|\W+', i)
        mer = [int(j) for j in mer]
        mer.sort()
        Peaks_df.loc[(Peaks_df['chr'] <= mer[-1]) & (Peaks_df['chr'] >= mer[0]), 'merge_chr'] = i

    return Peaks_df


def merge_(Peaks_num_stat, merge_thre):
    Peaks_num_stat = Peaks_num_stat[::-1]
    sum_ = 0
    split_loc = []
    n_chr = Peaks_num_stat.shape[0]
    for i in range(n_chr):
        sum_ = sum_ + Peaks_num_stat[Peaks_num_stat.index[i]]
        if sum_ > merge_thre:
            sum_ = 0
            split_loc.append(i + 1)

    # the first index
    if 0 in split_loc:
        del split_loc[0]

    # the last index
    if n_chr in split_loc:
        split_loc.pop()

    Peaks_num_stat_split = np.split(np.array(Peaks_num_stat), split_loc)

    Peaks_num_stat_merge = pd.DataFrame([Peaks_num_stat_split[i].sum() for i in range(len(Peaks_num_stat_split))],
                                        index=['_'.join(i.astype(str))
                                               for i in np.split(Peaks_num_stat.index, split_loc)],
                                        columns=['chr'])
    return Peaks_num_stat_merge[::-1]


def SCARP_cell_embedding(diffusion_mat, kept_comp, Cells):
    diffusion_mat = sp.csr_matrix(diffusion_mat)
    U, eig_value, V = svds(diffusion_mat, kept_comp)
    sort_index = abs(eig_value).argsort()[::-1]
    eig_value = eig_value[sort_index]
    U = U.T[sort_index].T
    svd_mat = U.dot(np.diag(eig_value))
    cell_mat = svd_mat / np.linalg.norm(svd_mat, axis=1).reshape(-1, 1)

    Cells_df = pd.DataFrame(cell_mat,
                            index=Cells,
                            columns=['feature' + str(kkk + 1) for kkk in range(cell_mat.shape[1])])
    return Cells_df


def SCARP_SD_plot(data, peaks_num=None, title=None,
                  log2=True, plot_SD=False,
                  fig_size=(4, 3), save_file=None):
    if peaks_num > 50000:
        max_k = 100
    else:
        max_k = 50

    U, eigvalue, V = svds(data, max_k)
    sort_index = abs(eigvalue).argsort()[::-1]
    eigvalue = eigvalue[sort_index]
    U = U.T[sort_index].T
    svd_mat = U.dot(np.diag(eigvalue))
    std_PCs = np.linalg.norm(svd_mat, axis=0)

    if log2:
        std_PCs = np.log2(std_PCs)

    std_PCs_var = [np.var(std_PCs[i:])
                   for i in range(len(std_PCs))] / np.var(std_PCs)
    kept_component = np.where(std_PCs_var > 5e-3)[0][-1] + 1

    # digit number chosen to be 0 or 5
    digit = kept_component % 10
    if digit <= 2:
        kept_component = kept_component - digit
    elif 3 <= digit <= 7:
        kept_component = kept_component - digit + 5
    elif digit > 7:
        kept_component = kept_component - digit + 10
    # print(f'kept_component: {kept_component}')

    if kept_component == 0:
        kept_component = 5

    # plot or not
    if plot_SD:
        plt.figure(figsize=fig_size)
        plt.plot(np.arange(1, max_k + 1), std_PCs, marker='o', markersize=5)
        plt.axvline(kept_component)
        plt.xlabel('Number of components')
        plt.ylabel('Standard Deviation')
        plt.title(title)
        if save_file is not None:
            plt.savefig(save_file)
            plt.close()

    return kept_component
