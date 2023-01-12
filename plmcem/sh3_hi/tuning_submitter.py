from itertools import product
import subprocess
import os

msa_file_name = 'sh3_hi_sampled_seqs.fa'#
tuning_batch_file = 'tuning.sh'

for folder in ['tuning', 'tuning/job_info', 'tuning/outputs']:
    if os.path.exists(folder):
        print(folder + ' exist, please make sure old runs are removed.')
    else:
        os.makedirs(folder)
        print(folder + ' created')

command_head = 'sbatch ' + tuning_batch_file + ' '

param_lists = [
    [1, 0], #gapignore
    [0.8, 2], #theta
    [2, 5, 10, 20], #maxiter
    [1, 0], #sgd/fast
    ['kmeans', 'random'] #initialization
]

#param_lists = [[1], [0.8], [2], [1], ['random']]

param_lists = [[str(num) for num in entry] for entry in param_lists]

for i, param_set in enumerate(product(*param_lists)):
    options = [str(i)] + list(param_set) + [msa_file_name]
    command = command_head + ' '.join(options)
    subprocess.run(command, shell = True, check = True, universal_newlines = True)
