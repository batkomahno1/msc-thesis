# THIS MODULE WILL:
# 1. FIND THE LAST COMPLETE ITERATION
# 2. CHECK THAT ALL SETS OF PARAMS ARE AT THE SAME ITERATION# 3. CONTINUE BUILDING CLN AND ADV GANS MID-ITERATION.
# 3. DETECT ADV SAMPLES
# NOTE: MODULE ASSUMES THAT GANS WHERE ALREADY CREATED!
# NOTE: CYCLE THROUGH ARCHITECTURES SO THAT ABLE TO STOP/RESUME ITERATIONS

import itertools
import time
import os
# THIS MUST HAPPEN BEFORE TORCH IS IMPORTED!!!
# makes pytorch cuda order GPUs the same way nvidis-smi does
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import pickle
import argparse
import logging
from collections import Counter
import shutil

from experiments import Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN

parser = argparse.ArgumentParser()
parser.add_argument("--nb_iter", type=int, default=1, help="number of iterations per experiment")
parser.add_argument("--verbose", type=lambda v: v=='True', default=False, help="verbose")
parser.add_argument("--test", type=lambda v: v=='True', default=False, help="measure runtimes")
parser.add_argument("--nb_gpus", type=int, default=4, help="number of gpus to be used")
parser.add_argument("--download", type=lambda v: v=='True', default=False, help="download weights")
parser.add_argument("--reset", type=lambda v: v=='True', default=False, help="redo detection")
opt = parser.parse_args()
print(opt)

# start logging
logging.basicConfig(filename='experiment.log', level=logging.INFO)
logging.info(f'RUNNING EXPERIMENT {opt}')
logging.info(time.asctime())

# find available GPUs and mark them for use
# always uses cuda:0
import os, torch, subprocess
device = 'cpu'
if opt.nb_gpus > 0 and torch.cuda.is_available():
    cmd = ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
    proc = subprocess.check_output(cmd)
    gpu_mem = [int(v) for v in proc.decode('utf-8').replace('\n', ' ').split()]
    nb_sys_gpus = torch.cuda.device_count()
    gpus_available = [0]
    # check if this is a multi-GPU machine
    if nb_sys_gpus > 1:
        gpus = range(nb_sys_gpus)
        gpus_available = [i for i in gpus if gpu_mem[i] < 4][:opt.nb_gpus]
    os.environ["CUDA_VISIBLE_DEVICES"]=','.join(str(i) for i in gpus_available)
    device = os.environ["CUDA_VISIBLE_DEVICES"][0]
    logging.info(f'Number GPUs: {len(gpus_available)}')
    print('CUDA_VISIBLE_DEVICES: ',os.environ["CUDA_VISIBLE_DEVICES"])

# Initialize architectures
var = [Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN]
val = ['wgan', 'wgan_gp', 'cgan', 'acgan']
GAN_CHOICES = {name:var[i] for i,name in enumerate(val)}

ITERATIONS = opt.nb_iter
if not opt.test:
    from config import *
    EXPERIMENTS = [GAN_CHOICES[name](epochs = GAN_SETTINGS[name][0], batch_size=GAN_SETTINGS[name][1], \
                    verbose=opt.verbose, device=device) for name in GAN_CHOICE]
else:
    opt.verbose=True
    from config_test import *
    EXPERIMENTS = [GAN_CHOICES[name](epochs = GAN_SETTINGS[name][0], batch_size=GAN_SETTINGS[name][1], \
                    verbose=opt.verbose, device=device) for name in GAN_CHOICE]

logging.info(f'Settings: {GAN_SETTINGS}')

ARCH_FAMILIES = {'WASSERSTEIN':('wgan', 'wgan_gp'), 'CONDITIONAL':('cgan', 'acgan')}
PARAM_SET = {}
RUN_NAME = 'detection_'+'_'.join(time.asctime().split(' ')[1:3]).lower()+'_'+time.asctime().split(' ')[-1]
RES_DIR =  os.getcwd() + '/experiment_results/'
RUN_PATH = RES_DIR + RUN_NAME + '.pkl'
RUN_PATH_CURR = RES_DIR + 'detection.pkl'

if opt.reset:
    for f in [RUN_PATH, RUN_PATH_CURR]:
        if os.path.isfile(f): os.remove(f)

if not os.path.exists(RES_DIR): os.makedirs(RES_DIR)

def get_params(*args):
    return list(itertools.product(*args))

def save_res(obj):
    with open(RUN_PATH, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    with open(RUN_PATH_CURR, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# set different target classes and target epochs for conditional and wasserstein architectures
# tgt_class, tgt_epochs = (-1,), (0,)
tgt_class = (-1,)
PARAM_SET['WASSERSTEIN'] = get_params(tgt_class, TGT_EPOCHS, PCT, EPS, TARGETED, DATASET, ATK, NOTE)

tgt_class = (6,)
PARAM_SET['CONDITIONAL'] = get_params(tgt_class, TGT_EPOCHS, PCT, EPS, TARGETED, DATASET, ATK, NOTE)

if opt.verbose: print('Starting runs...')

# helper method
from itertools import groupby
def all_equal(iterable):
    "Returns True if all the elements are equal to each other"
    "Source: python doc"
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

# find last iterarion
# it will overwrite half finished ITERATIONS
# consistency check: every parameter set was run the same number of times
iter_start = 0
if os.path.isfile(RUN_PATH_CURR):
    with open(RUN_PATH_CURR, 'rb') as f:
        var = pickle.load(f)
        # list of iterations ie. [0,0,0,1,1,1,...] from result.pkl
        iter_list = [k[-1] for k in var.keys()]
        # experiments per iteration
        iter_count = Counter(iter_list)
        if not all_equal(iter_count.values()):
            raise RuntimeError('Inconsistent num. of iterations per experiment!', iter_count)
        iter_start = max(iter_list) + 1
        if opt.verbose: print('Resuming at iteration', iter_start)

# run the exp
result = dict()
for itr in range(iter_start, iter_start + ITERATIONS):
    for exp in EXPERIMENTS:
        gan_name = exp.GAN_NAME
        arch_family = [k for k,v in ARCH_FAMILIES.items() if gan_name in v][0]
        # run detection
        for params in PARAM_SET[arch_family]:
            if opt.verbose: print(gan_name, params, itr)
            eps, note = params[3], params[-1]
            if 'downgrade'==note.lower() and eps==0.0:
                logging.info(f'Experiment {params} skipped!.')
                continue
            start = time.time()
            auc, fprs, tprs, thetas = exp.detect(params, itr=itr, download=opt.download)
            result[(gan_name, params, itr)] = auc, fprs, tprs, thetas, time.time()-start
    # save at the end of an iteration
    save_res(result)
