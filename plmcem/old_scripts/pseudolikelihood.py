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
import matplotlib.pyplot as plt

from evcouplings.couplings import CouplingsModel
from evcouplings.couplings.model import _single_mutant_hamiltonians
from bioviper import MultipleSequenceAlignment as MSA
from bioviper import readAlignment

parser = argparse.ArgumentParser(description="Pseudolikelihood Parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('msa', help='Input alignment file name.')
parser.add_argument('params', help='Input param file name.')

args = parser.parse_args()
config = vars(args)

alphabet = 'ACDEFGHIKLMNPQRSTVWY'
the_rest = 'BJOUXZ'
gap = '-'

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
    
def pos_I(msa_obj):
    '''
    Tensor of ones and zeros, each matrix (going through the first dimension) represents one position
    '''
    if not hasattr(msa_obj, 'positioned_identity'):
        L = msa_obj.L
        alphabet_size = 21 - msa_obj.gap_ignore
        ans = np.zeros((L, alphabet_size, alphabet_size*L))
#        ans = torch.zeros(L, alphabet_size, alphabet_size*L)
        for i in range(L):
            ans[i, range(20), range(20*i, 20*(i+1))] = 1
#            ans[i, torch.arange(alphabet_size), alphabet_size*i + torch.arange(alphabet_size)] = 1
        msa_obj.positioned_identity = torch.tensor(ans, device = msa_obj.device, dtype = torch.float32)
#        msa.positioned_identity = ans.to(msa.device)
    return msa_obj.positioned_identity

def pll(msa, hi, Jij):
    '''
    Calculate pseudolikelihood of each sequence in the MSA
    '''
    #calculate hamiltonian contribution of each position and possible amino acid
    #UI = torch.matmul(msa.pos_I(), hi) + torch.matmul(torch.matmul(msa.pos_I(), Jij), msa.onehot_msa())
    
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
MSA.pos_I = pos_I
MSA.pll = pll

def load_params(param_file_name, device):
    '''
    Get plmc output parameters for each cluster
    '''
    model = CouplingsModel(param_file_name)
    params = (hi_to_onehot(model.h_i, device), Jij_to_onehot(model.J_ij, device))
    return params

def hi_to_onehot(hi, device):
    '''
    Convert 2D (L x alphabet_size) h_i (fields) matrix to 2D (L*alphabet_size x 1) using onehot
    '''
    return torch.tensor(np.expand_dims(hi, -1), device = device, dtype = torch.float32)

def Jij_to_onehot(Jij, device):
    '''
    Convert 4D (L x L x alphabet_size x alphabet_size) J_ij (pairwise terms) matrix to 2D (L*alphabet_size x L*alphabet_size) using onehot
    '''
    return torch.tensor(Jij.swapaxes(2, 1).reshape(Jij.shape[0], Jij.shape[-1], Jij.shape[0] * Jij.shape[-1]), device = device, dtype = torch.float32)
msa = readAlignment(config['msa'])
msa.gap_ignore = 1
msa.device = torch.device('cpu')
params = load_params(config['params'], msa.device)
print(float(torch.mean(pll(msa, *params))))
