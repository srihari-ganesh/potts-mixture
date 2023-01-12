import collections.abc
collections.Iterable = collections.abc.Iterable #Issue in evcouplings I think - change in structure
import os
import argparse
from datetime import datetime
import subprocess
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch import nn
from tabulate import tabulate
import sklearn.cluster as skcluster
from evcouplings.couplings import CouplingsModel
from alignmenttools.MultipleSequenceAlignment import MultipleSequenceAlignment as MSA
from alignmenttools.MultipleSequenceAlignment import readAlignment
from itertools import product

msa_file_name = 'sh3_hi_sampled_seqs.fa'
true_cluster_file_names = ['sh3_sampled_seqs.fa', 'sh3_hi_pdz_Jij_sampled_seqs.fa']

def main(msa_file_name, true_cluster_file_names):
    msa_file_name_head = msa_file_name.split('.')[0]
    true_cluster_ids = [set(readAlignment(file_name).ids) for file_name in true_cluster_file_names]
    
    param_lists = [
        [1, 0], #gapignore
        [0.8, 2], #theta
        [2, 5, 10, 20], #maxiter
        [1, 0], #sgd/fast
        ['kmeans', 'random'] #initialization
    ]
    file_name_head = 'tuning/outputs/tuning_run_'

    columns = ['gap_ignore', 'theta', 'maxiter', 'fast', 'initialization', 'em iterations', 'i0:t0', 'i0:t1', 'i1:t0', 'i1:t1', 'score']
    df_dict = {column: [] for column in columns}
#    df_dict = {column: [] for column in columns[:6]}

    for i, param_set in enumerate(product(*param_lists)):
        folder_name = file_name_head + str(i) + '/output/'
        for j, param in enumerate(columns[:5]):
            df_dict[param].append(param_set[j])
        inferred_cluster_file_names = [file_name_head + str(i) + '/output/' + msa_file_name_head + '_cluster' + str(k) + '.fa' for k in range(2)]
        if np.all([os.path.exists(file_name) for file_name in inferred_cluster_file_names]):
            mat, score = compare(inferred_cluster_file_names, true_cluster_ids)
            for inferred_cluster_ind in range(2):
                for true_cluster_ind in range(2):
                    df_dict['i' + str(inferred_cluster_ind) + ':t' + str(true_cluster_ind)].append(mat[inferred_cluster_ind][true_cluster_ind])
            df_dict['score'].append(round(score, 3))
            #print(mat, score)
        else:
            for inferred_cluster_ind in range(2):
                for true_cluster_ind in range(2):
                    df_dict['i' + str(inferred_cluster_ind) + ':t' + str(true_cluster_ind)].append(np.nan)
            df_dict['score'].append(np.nan)
            #print('Run failed')
        
        iter_ind = 0
        while os.path.exists(file_name_head + str(i) + '/iterations/' + msa_file_name_head + '_cluster0_iter' + str(iter_ind) + '.fa'):
            iter_ind += 1
        df_dict['em iterations'].append(iter_ind - 1)

    df = pd.DataFrame(df_dict)
    df = df.sort_values('score', ascending = False)
    open('tuning/tuning_analysis.txt', 'w').write(tabulate(df, headers = 'keys'))

def compare(cluster_file_names, true_cluster_ids):
    cluster_ids = [set(readAlignment(file_name).ids) for file_name in cluster_file_names]
    overlaps = [[len(tcis.intersection(cis)) for tcis in true_cluster_ids] for cis in cluster_ids]
    
    for i in range(2):
        for j in range(2):
            if overlaps[i][j] == 1:
                print(cluster_ids[i].intersection(true_cluster_ids[j]))

    max_score = [len(t) for t in true_cluster_ids]
    max_score = sum([t*(t-1)/2 for t in max_score])
    actual_score = sum([sum([val*(val-1)/2 - sum([val1*val for val1 in row[i+1:]]) for i, val in enumerate(row)]) for row in overlaps])
    return overlaps, 1.0 * actual_score/max_score

#def compare(cluster_names, cluster_file_names, true_cluster_names, true_cluster_ids, write_file = False, first = False):
#    cluster_ids = [set(readAlignment(cluster_file_name).ids) for cluster_file_name in cluster_file_names]
#    overlaps = [[len(tcis.intersection(cis)) for tcis in true_cluster_ids] for cis in cluster_ids]
#    content = tabulate(overlaps, true_cluster_names, showindex = [cluster_name + '.fa' for cluster_name in cluster_names])
#    max_score = [len(t) for t in true_cluster_ids]
#    max_score = sum([t*(t-1)/2 for t in max_score])
#    actual_score = sum([sum([val*(val-1)/2 - sum([val1 * val for val1 in row[i+1:]]) for i, val in enumerate(row)]) for row in overlaps])
#    cluster_str = '\n'*2 + 'Clustering score (max 1.0): ' + str(1.0 * actual_score/max_score) + '\n'*2
#    if first:
#        mode = 'w'
#    else:
#        mode = 'a'
#    if write_file:
#        open(write_file, mode).write(content + cluster_str)
#    return content + cluster_str

main(msa_file_name, true_cluster_file_names)
