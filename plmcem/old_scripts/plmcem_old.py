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
from bioviper import MultipleSequenceAlignment as MSA
from bioviper import readAlignment

ints = ['em_maxiter', 'em_numclusters', 'gapignore', 'ncores', 'fast', 'em_no_iter_print', 'em_final_print']
plmc_args = [('scale', '-s'), ('theta', '-t'), ('lambdah', '-lh'), ('lambdae', '-le'), ('lambdag', '-lg'), ('maxiter', '-m')]

parser = argparse.ArgumentParser(description="PLMC EM Parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('msa-file-name', help='Input alignment file name.')
parser.add_argument('-em', '--em-maxiter', default=100, help='Maximum number of iterations of the EM algorithm.')
parser.add_argument('-ek', '--em-numclusters', default=2, help='Number of clusters.')
parser.add_argument('-ed', '--em-device', help='Processing unit to use PyTorch on; use PyTorch name (CPU = "cpu", GPU = "cuda", etc.).')
parser.add_argument('-ei', '--em-init', default='random', help='Method to use for initializing cluster assignments. Valid inputs are "random" and "kmeans", with random as the default.')
parser.add_argument('-enip', '--em-no-iter-print', action='store_true', help='Do not print PLMC iteration outputs (and true cluster comparison, if applicable).')
parser.add_argument('-efp', '--em-final-print', action='store_true', help='Print final cluster comparison (if applicable).')
parser.add_argument('-eht', '--em-hyper-tune', default=None, help='Run a hyperparameter tuning run.')

parser.add_argument('-fp', '--plmc', default='../../plmc/bin/plmc', help='Path to PLMC binary, can be relative to script location.')
parser.add_argument('-do', '--output', default='', help='Output directory.')
parser.add_argument('-fc', '--true-clusters', help='File containing names of alignment files consisting of the true clusters of the MSA.')

parser.add_argument('-sgd', '--fast', action='store_true', help='PLMC general: Fast weights and stochastic gradient descent.')
parser.add_argument('-s', '--scale', help='PLMC alignment processing: Sequence weights: neighborhood weight [s > 0].')
parser.add_argument('-t', '--theta', help='PLMC alignment processing: Sequence weights: neighborhood divergence [0 < t < 1].')
parser.add_argument('-lh', '--lambdah', default='0.01', help='PLMC Maximum a posteriori estimation (L-BFGS, default): Set L2 lambda for fields (h_i).')
parser.add_argument('-le', '--lambdae', default='16.0', help='PLMC Maximum a posteriori estimation (L-BFGS, default): Set L2 lambda for couplings (e_ij). Note that the maximum is typically 100.0')
parser.add_argument('-lg', '--lambdag', help='PLMC Maximum a posteriori estimation (L-BFGS, default): Set group L1 lambda for couplings (e_ij).')
parser.add_argument("-g", '--gapignore', action='store_true', help='PLMC general: Exclude first alphabet character in potential calculations.')
parser.add_argument("-m", '--maxiter', default='100', help='PLMC general: Maximum number of iterations. Note that PLMC default is typically 0 (no maximum).')
parser.add_argument('-n', '--ncores', default=mp.cpu_count(), help='PLMC general: Maximum number of threads to use in OpenMP')

args = parser.parse_args()
config = vars(args)

#print(config)

for arg in ints:
    config[arg] = int(config[arg])

msa_file_name = config['msa-file-name']
num_iters = config['em_maxiter']
num_clusters = config['em_numclusters']
device_name = config['em_device']
cluster_init = config['em_init']
if cluster_init != 'random' and cluster_init != 'kmeans':
    cluster_init = 'random'
gap_ignore = config['gapignore']

em_iter_print = 1 - config['em_no_iter_print']
em_final_print = config['em_final_print']

plmc_path = config['plmc']
output_dir = config['output']
#may have compatability issues if backslashes used
if len(output_dir) > 0 and output_dir[-1] != '/':
    output_dir += '/'
iterations_dir = output_dir + 'iterations/'
final_dir = output_dir + 'output/'
for folder in [output_dir, iterations_dir, final_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        if em_iter_print:
            print(folder, 'created')

true_clusters = config['true_clusters']

if em_hyper_tune != None:
    i = int(em_hyper_tune)
    m_arr = [2, 5, 10, 25, 100]
    g_arr = [True, False]
    fast_arr = [True, False]
    
    m_i, g_i, fast_i = np.unravel_index(i, (len(m_arr), len(g_arr), len(fast_arr)))

    config['maxiter'] = m_arr[m_i]
    gap_ignore = g_arr[g_i]
    config['fast'] = fast_arr[fast_i]

plmc_options = []
for arg in plmc_args:
    if config[arg[0]] != None:
        plmc_options.append(arg[1]  + ' ' + config[arg[0]])

#plmc_options.append('-n ' + str(int(config['ncores']/num_clusters)))
plmc_options.append('-n ' + str(config['ncores']))

if gap_ignore:
    plmc_options.append('-g')

if config['fast']:
    plmc_options.append('--fast')

tester = torch.tensor(np.array([1]))
if device_name:
    try:
        device = torch.device(device_name)
        tester.to(device)
    except:
        try:
            device = torch.device('cuda')
            tester.to(device)
        except:
            device = torch.device('cpu')
            tester.to(device)
else:
    try:
        device = torch.device('cuda')
        tester.to(device)
    except:
        device = torch.device('cpu')
        tester.to(device)

if em_iter_print:
    print(str(config) + '\n')
    print('Using ' + device + '\n')

rng = np.random.default_rng()

def read_fasta(name, form = '.fa'):
    '''
    Read alignment file into MSA object
    '''
    return readAlignment(name + '.' + form)

def kmeans_assignments(msa, num_clusters):
    '''
    Given MSA object, returns a list of assignments that results from kmeans clustering.
    '''
    result = skcluster.KMeans(n_clusters = num_clusters).fit(msa.onehot_msa().T)
    assignment_list = list(result.labels_)
    assignments = [[] for i in range(num_clusters)]
    for j in range(msa.N):
        assignments[assignment_list[j]].append(j)
    return assignments, assignment_list

def random_assignments(msa, num_clusters):
    '''
    Given MSA object, returns a list of assignments. Needs to be able to get number of sequences in MSA object
    '''
    if type(msa) == MSA:
        n = msa.N
    else:
        n = msa
#    swap_num = 2200
#    c1 = 11658
#    c2 = 29578
#    assignments = [list(range(c1 - swap_num)) + list(range(c1, c1+swap_num)), list(range(c1-swap_num, c1)) + list(range(c1+swap_num, c2))]
#    assignments = [list(range(int(n/num_clusters*i), int(n/num_clusters*(i+1)))) for i in range(num_clusters)]
#    permutation = list(range(num_clusters))(*int(n/num_clusters)+1)[:n]    
    permutation = rng.permutation(n)
    assignments = [permutation[int(n/num_clusters*i):int(n/num_clusters*(i+1))] for i in range(num_clusters)]
    assignment_list = []
    for i in range(num_clusters):
        assignment_list += [i] * (int(n/num_clusters*(i+1)) - int(n/num_clusters*i))
    return assignments, assignment_list

def get_cluster_names(msa_name, num_clusters, iteration = None):
    '''
    Get cluster names (without file extension) using naming convention with cluster ID and iteration number
    '''
    msa_name = msa_name.split('/')[-1]
    msa_name = msa_name.split('\\')[-1]
    cluster_ids = list(range(num_clusters))
    if iteration == None:
        cluster_names = [msa_name + '_cluster' + str(i) for i in cluster_ids]
    else:
        cluster_names = [msa_name + '_cluster' + str(i) + '_iter' + str(iteration) for i in cluster_ids]
    return cluster_names

def split_msa(msa, assignments, cluster_names):
    '''
    Given MSA object and cluster assignment of each sequence, returns separate MSA objects for each cluster
    '''
    cluster_msas = [MSA([msa._records[i] for i in assignment]) for assignment in assignments]
    for cluster_name, cluster_msa in zip(cluster_names, cluster_msas):
        write_fasta(cluster_msa, iterations_dir + cluster_name)

def write_fasta(msa_obj, name):
    '''
    Writes msa into fasta file of given name
    '''
    msa_obj.save(name + '.fa')

def initialize(msa_file_name, num_clusters, cluster_init, gap_ignore, device):
    msa_name, msa_format = msa_file_name.split('.')
    msa = read_fasta(msa_name, msa_format)
    msa.gap_ignore = gap_ignore
    msa.device = device
    cluster_names = get_cluster_names(msa_name, num_clusters, 0)
    if cluster_init == 'kmeans':
        assignment_method = kmeans_assignments
    else:
        assignment_method = random_assignments
    assignments, assignment_list = assignment_method(msa, num_clusters)
    split_msa(msa, assignments, cluster_names)
    return msa, msa_name, cluster_names, assignment_list

def run_plmc(cluster_name):
    #choosing not to have focus by default for whatever reason
    '''
    Run plmc from command line for a given cluster
    '''
    #currently taking advantage of raw command line but full path is below if needed
    #plmc_location = '/'.join(__file__.split('/')[:-2]) + '/plmc/bin/plmc'
    #What about run_plmc on python
    #print('\n' + cluster_name + ':' + '\n')
    command = [plmc_path, '-o ' + iterations_dir + cluster_name + '.params', *plmc_options, iterations_dir + cluster_name + '.fa']
    #print(command)
    ans = subprocess.run(' '.join(command), shell = True, check = True, capture_output = True, universal_newlines = True).stderr
    return cluster_name, ans

def run_plmcs(cluster_names):
    '''
    Running plmc on each of the clusters
    '''
    if __name__ == '__main__' and config['ncores'] > 1:
        pool = mp.Pool(processes = max(min(mp.cpu_count()-1, len(cluster_names)), 1))
        ans = pool.map_async(run_plmc, cluster_names).get()
        pool.close()
        pool.join()
        ans = {x[0]: x[1] for x in ans}
        ans = [ans[cluster_name] for cluster_name in cluster_names]
        if em_iter_print:    
            for string, cluster_name in zip(ans, cluster_names):
                print('\n' + cluster_name + ':\n')
                print(string)
    else:
        outs = [run_plmc(cluster_name) for cluster_name in cluster_names]
        if em_iter_print:
            for out in outs:
                print('\n' + out[0]  + ':\n')
                print(out[1])

    return [iterations_dir + cluster_name + '.params' for cluster_name in cluster_names]

alphabet = 'ACDEFGHIKLMNPQRSTVWY'
the_rest = 'BJOUXZ'
gap = '-'
gap_alt = '.'

onehot_dicts = [{}]*2

#Making dictionaries for conversion from letters to one-hot dictionaries
#if treating gaps as characters
onehot_dicts[0] = {letter: ([0] * (len(alphabet)+1)) for letter in the_rest}
for i, letter in enumerate(gap + alphabet): #check to see where gaps come in the hi, Jij matrix - I believe it's at the start
    onehot_dicts[0][letter] = [0] * i + [1] + [0] * (len(alphabet) - i)

#if ignoring gaps
onehot_dicts[1] = {letter: ([0] * len(alphabet)) for letter in the_rest + gap}
for i, letter in enumerate(alphabet):
    onehot_dicts[1][letter] = [0] * i + [1] + [0] * (len(alphabet) - i - 1)

for onehot_dict in onehot_dicts:
    for letter in alphabet + the_rest:
        onehot_dict[letter.lower()] = onehot_dict[letter]
    for char in gap_alt:
        onehot_dict[gap_alt] = onehot_dict[gap]
def onehot_seqs(msa_obj):
    '''
    Convert letter sequences to onehot
    '''
    onehot_dict = onehot_dicts[msa_obj.gap_ignore]
    full_ans = [[] for seq in msa_obj.matrix]
    for i, seq in enumerate(msa_obj.matrix):
        for letter in seq:
            full_ans[i] += onehot_dict[letter]
#    for record in msa_obj._records:
#        onehot_dict = onehot_dicts[msa_obj.gap_ignore]
#        ans = []
#        for letter in record.seq:
#            ans += onehot_dict[letter]
#        full_ans.append(ans)
    return full_ans

def onehot_init(msa_obj):
    '''
    Create all reusable tensors needed for pseudolikelihood calculations
    '''
    oh = torch.tensor(np.array(msa_obj.onehot_seqs()), dtype = torch.float32).T
    msa_obj.pll_onehot = oh.to(msa_obj.device)
    msa_obj.pll_onehot_3d = msa_obj.pll_onehot.reshape(msa_obj.L, -1, msa_obj.N)

def onehot_msa(msa_obj):
    '''
    MSA represented as concatenated one-hot tensor (in a sequence, each residue's onehot vector is concatenated)
    '''
    if not hasattr(msa_obj, 'pll_onehot'):
        msa_obj.onehot_init()
    return msa_obj.pll_onehot

def onehot_3d_msa(msa_obj):
    '''
    MSA represented as standard one-hot tensor (new dimension added)
    '''
    if not hasattr(msa_obj, 'pll_onehot_3d'):
        msa_obj.onehot_init()
    return msa_obj.pll_onehot_3d
    
#def pos_I(msa_obj):
#    '''
#    Tensor of ones and zeros, each matrix (going through the first dimension) represents one position
#    '''
#    if not hasattr(msa_obj, 'positioned_identity'):
#        L = msa_obj.L
#        alphabet_size = 21 - msa_obj.gap_ignore
#        ans = np.zeros((L, alphabet_size, alphabet_size*L))
##        ans = torch.zeros(L, alphabet_size, alphabet_size*L)
#        for i in range(L):
#            ans[i, range(20), range(20*i, 20*(i+1))] = 1
##            ans[i, torch.arange(alphabet_size), alphabet_size*i + torch.arange(alphabet_size)] = 1
#        msa_obj.positioned_identity = torch.tensor(ans, device = msa_obj.device, dtype = torch.float32)
##        msa.positioned_identity = ans.to(msa.device)
#    return msa_obj.positioned_identity

def pll(msa, hi, Jij):
    '''
    Calculate pseudolikelihood of each sequence in the MSA
    '''
    #calculate hamiltonian contribution of each position and possible amino acid
#    UI = torch.matmul(msa.pos_I(), hi) + torch.matmul(torch.matmul(msa.pos_I(), Jij), msa.onehot_msa())
    UI = hi + torch.matmul(Jij, msa.onehot_msa())
    #normalize log probabilities across amino acids at each position
    ll_energies = nn.functional.log_softmax(UI, dim = 1)
    #sum hamiltonian contributions for each residue in a sequence to get the pseudolikelihood; do for all sequences
    ans = torch.sum(torch.mul(ll_energies, msa.onehot_3d_msa()), dim = (0, 1))
    return ans

#Turn defined methods into part of the MSA class
MSA.onehot_seqs = onehot_seqs
MSA.onehot_init = onehot_init
MSA.onehot_msa = onehot_msa
MSA.onehot_3d_msa = onehot_3d_msa
#MSA.pos_I = pos_I
MSA.pll = pll

def load_params(param_file_names, device):
    '''
    Get plmc output parameters for each cluster
    '''
    models = [CouplingsModel(param_file_name) for param_file_name in param_file_names]
    params = [(hi_to_onehot(model.h_i, device), Jij_to_onehot(model.J_ij, device)) for model in models]
    return params

def hi_to_onehot(hi, device):
    '''
    Convert 2D (L x alphabet_size) h_i (fields) matrix to 2D (L*alphabet_size x 1) using onehot
    '''
    return torch.tensor(np.expand_dims(hi, -1), device = device, dtype = torch.float32)
#    return torch.tensor(hi.reshape(-1, 1), device = device, dtype = torch.float32)

def Jij_to_onehot(Jij, device):
    '''
    Convert 4D (L x L x alphabet_size x alphabet_size) J_ij (pairwise terms) matrix to 2D (L*alphabet_size x L*alphabet_size) using onehot
    '''
    return torch.tensor(Jij.swapaxes(2, 1).reshape(Jij.shape[0], Jij.shape[-1], Jij.shape[0] * Jij.shape[-1]), device = device, dtype = torch.float32)
#    return torch.tensor(Jij.swapaxes(2, 1).reshape(Jij.shape[0] * Jij.shape[-1], Jij.shape[0] * Jij.shape[-1]), device = device, dtype = torch.float32)

def get_assignments(msa, param_tuples, num_clusters):
    '''
    Use pseudolikelihoods to assign sequences to new clusters (sequence -> cluster which gave maximum pseudolikelihood)
    '''
    plls = torch.stack([msa.pll(*param_tuple) for param_tuple in param_tuples])
#    print(plls[:, 0])
    maxes, cluster_assignments = torch.max(plls, dim = 0)

    #for each/any empty clusters, assign like 50 sequences with the worst maximum pseudolikelihood (maximum taken among clusters) to their own new cluster of 50 (in hopes of getting better parameters?)
    for i in range(num_clusters):
        if i not in cluster_assignments:
            for k in range(50):
                j = torch.argmin(maxes)
                cluster_assignments[j] = i
                maxes[j] = torch.max(maxes) + 1
    cluster_inds = [[] for i in range(num_clusters)]
    for i, cluster_assignment in enumerate(cluster_assignments):
        cluster_inds[cluster_assignment].append(i)
    return cluster_inds, cluster_assignments

def reassign(msa, param_file_names, msa_name, num_clusters, iteration):
    param_tuples = load_params(param_file_names, msa.device)
    assignments, assignment_list = get_assignments(msa, param_tuples, num_clusters)
    cluster_names = get_cluster_names(msa_name, num_clusters, iteration)
    split_msa(msa, assignments, cluster_names)
    return cluster_names, list(assignment_list)

def compare(cluster_names, cluster_file_names, true_cluster_names, true_cluster_ids, write_file = False, first = False):
    cluster_ids = [set(readAlignment(cluster_file_name).ids) for cluster_file_name in cluster_file_names]
    overlaps = [[len(tcis.intersection(cis)) for tcis in true_cluster_ids] for cis in cluster_ids]
    content = tabulate(overlaps, true_cluster_names, showindex = [cluster_name + '.fa' for cluster_name in cluster_names])

    max_score = [len(t) for t in true_cluster_ids]
    max_score = sum([t*(t-1)/2 for t in max_score])
    actual_score = sum([sum([val*(val-1)/2 - sum([val1 * val for val1 in row[i+1:]]) for i, val in enumerate(row)]) for row in overlaps])
    cluster_str = '\n'*2 + 'Clustering score (max 1.0): ' + str(1.0 * actual_score/max_score) + '\n'*2

    if first:
        mode = 'w'
    else:
        mode = 'a'
    if write_file:
        open(write_file, mode).write(content + cluster_str)
    return content + cluster_str

def time(name, t1, t0):
    return name + ': ' + str(t1-t0) + '\n'

dtn = datetime.now

def run_em(msa_file_name, num_iters, num_clusters, cluster_init, gap_ignore, device, true_clusters, output_dir):
    header = msa_file_name.split('/')[-1]
    header = header.split('\\')[-1]
    header = header.split('.')[0] + '_cluster'
    info_file_name = output_dir + header +  '_info.txt'
    t0 = dtn()
    with open(info_file_name, 'w') as f:
        f.write(str(config)+'\n'*2)
        t = dtn()
        msa, msa_name, cluster_names, prev_assignment_list = initialize(msa_file_name, num_clusters, cluster_init, gap_ignore, device)
        f.write(time('init', dtn(), t))
        t = dtn()
        if em_iter_print:
            print('\nIteration 0:')
        param_file_names = run_plmcs(cluster_names)
        f.write(time('plmc 0', dtn(), t))
        if true_clusters:
            t = dtn()
            with open(true_clusters, 'r') as tcf:
                true_cluster_file_names = tcf.readlines()
                true_cluster_file_names = [line.rstrip() for line in true_cluster_file_names] 
            true_cluster_names = [true_cluster_file_name.split('/')[-1] for true_cluster_file_name in true_cluster_file_names]
            true_cluster_names = [true_cluster_file_name.split('\\')[-1] for true_cluster_file_name in true_cluster_file_names]
            true_cluster_ids = [set(readAlignment(true_cluster_file_name).ids) for true_cluster_file_name in true_cluster_file_names]
            f.write(time('true cluster comparison initialization', dtn(), t))
            
            t = dtn()
            cluster_file_names = [iterations_dir + cluster_name + '.fa' for cluster_name in cluster_names]
            compare_output = compare(cluster_names, cluster_file_names, true_cluster_names, true_cluster_ids, write_file = iterations_dir + header + '_iterations_comparison.txt', first = True)
            f.write(time('comparison 0', dtn(), t))
            if em_iter_print:
                print(compare_output)
         
        for i in range(1, num_iters+1):
            if em_iter_print:
                print('\nIteration ' + str(i) + ':')
            t = dtn()
            cluster_names, assignment_list = reassign(msa, param_file_names, msa_name, num_clusters, i)
            f.write(time('cluster ' + str(i), dtn(), t))

            t = dtn()
            cluster_file_names = [iterations_dir + cluster_name + '.fa' for cluster_name in cluster_names]
            if true_clusters:
                compare_output = compare(cluster_names, cluster_file_names, true_cluster_names, true_cluster_ids, write_file = iterations_dir + header + '_iterations_comparison.txt')
                f.write(time('comparison ' + str(i), dtn(), t))
                if em_iter_print:
                    print(compare_output)
            if assignment_list == prev_assignment_list:
                if em_iter_print:
                    print('TERMINATING: Repeated cluster assignments')
                break
            else:
                prev_assignment_list = assignment_list

            t = dtn()
            param_file_names = run_plmcs(cluster_names)
            f.write(time('plmc ' + str(i), dtn(), t))
        
        t = dtn()
        new_cluster_names = ['_'.join(cluster_name.split('_')[:-1]) for cluster_name in cluster_names]        
        for (cluster_name, new_cluster_name, param_file_name) in zip(cluster_names, new_cluster_names, param_file_names):
            subprocess.check_call(' '.join(['cp', iterations_dir + cluster_name + '.fa', output_dir + 'output/' + new_cluster_name + '.fa']), shell = True)
            subprocess.check_call(' '.join(['cp', param_file_name, final_dir + new_cluster_name + '.params']), shell = True)
        f.write(time('copy last iter files', dtn(), t))
        cluster_names = new_cluster_names

        f.write(time('total EM run time', dtn(), t0))
        if i < num_iters:
            f.write('termination reason: repeated cluster assignments\n')
            f.write(str(i) + ' iterations\n')
        else:
            f.write('termination reason: maximum iterations reached\n')
            f.write(str(num_iters) +  ' iterations\n')
        if true_clusters:
            t = dtn()
            cluster_file_names = [final_dir + cluster_name + '.fa' for cluster_name in cluster_names]
            comp = compare(cluster_names, cluster_file_names, true_cluster_names, true_cluster_ids)
            if em_final_print:
                print(comp)
            open(output_dir + header + '_comparison.txt', 'w').write(comp)
            f.write(time('final comparison', dtn(), t)+'\n')
            f.write(comp)

run_em(msa_file_name, num_iters = num_iters, num_clusters = num_clusters, cluster_init = cluster_init, gap_ignore = gap_ignore, device = device, true_clusters = true_clusters, output_dir = output_dir)
