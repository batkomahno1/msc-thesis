# THIS MODULE WILL:
# 1. FIND THE LAST ITERATION
# 2. CHECK THAT ALL SETS OF PARAMS ARE AT THE SAME ITERATION
# 3. CONTINUE BUILDING CLN AND ADV GANS MID-ITERATION.
# 4. MEASURE FIDs FROM SCRATCH
# NOTE: TO BUILD GANS FROM SCRATCH ONE MUST USE THE RESET OPTION
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
from scipy.stats import truncnorm
import copy
from experiments import Experiment_DPWGAN, Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN

parser = argparse.ArgumentParser()
parser.add_argument("--nb_iter", type=int, default=1, help="number of iterations per experiment")
parser.add_argument("--verbose", type=lambda v: v=='True', default=False, help="verbose")
parser.add_argument("--test", type=lambda v: v=='True', default=False, help="measure runtimes")
parser.add_argument("--nb_gpus", type=int, default=4, help="number of gpus to be used")
parser.add_argument("--download", type=lambda v: v=='True', default=False, help="download weights")
parser.add_argument("--reset", type=lambda v: v=='True', default=False, help="redo everything")
parser.add_argument("--soft_reset", type=lambda v: v=='True', default=False, help="redo reports")
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
# TODO: put this somewhere in a config file
var = [Experiment_DPWGAN, Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN]
val = ['dpwgan', 'wgan', 'wgan_gp', 'cgan', 'acgan']
GAN_CHOICES = {name:var[i] for i,name in enumerate(val)}

ITERATIONS = opt.nb_iter
if not opt.test:
    from config_dp import *
    EXPERIMENTS = [GAN_CHOICES[name](epochs = GAN_SETTINGS[name][0], batch_size=GAN_SETTINGS[name][1], \
                    verbose=opt.verbose, device=device) for name in GAN_CHOICE]
else:
    opt.verbose=True
    from config_test_dp import *
    EXPERIMENTS = [GAN_CHOICES[name](epochs = GAN_SETTINGS[name][0], batch_size=GAN_SETTINGS[name][1], \
                    verbose=opt.verbose, device=device) for name in GAN_CHOICE]

logging.info(f'Settings: {GAN_SETTINGS}')

ARCH_FAMILIES = {'WASSERSTEIN':('dpwgan', 'wgan', 'wgan_gp'), 'CONDITIONAL':('cgan', 'acgan')}
PARAM_SET = {}
RUN_NAME = 'run_'+'_'.join(time.asctime().split(' ')[1:3]).lower()+'_'+time.asctime().split(' ')[-1]
RES_DIR =  os.getcwd() + '/experiment_results/'
RUN_PATH_CURR = RES_DIR + 'results_dp.pkl'

# delete all old data
if opt.reset:
    if os.path.exists(RES_DIR):
        shutil.rmtree(RES_DIR)
    for exp in EXPERIMENTS:
        exp.reset()

# create results dir if necessary
if not os.path.exists(RES_DIR): os.makedirs(RES_DIR)

# delete reports BUT not the weights nor adv samples
if opt.soft_reset:
    if os.path.isfile(RUN_PATH_CURR):
        os.remove(RUN_PATH_CURR)
    for exp in EXPERIMENTS:
        exp.soft_reset()

def get_params(*args):
    return list(itertools.product(*args))

def save_res(obj):
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
        result = pickle.load(f)
        # list of iterations ie. [0,0,0,1,1,1,...] from result.pkl
        iter_list = [k[-1] for k in result.keys()]
        # experiments per iteration
        iter_count = Counter(iter_list)
        if not all_equal(iter_count.values()):
            raise RuntimeError('Inconsistent num. of iterations per experiment!', iter_count)
        iter_start = max(iter_list) + 1
        if opt.verbose: print('Resuming at iteration', iter_start)
else:
    result = dict()

# run the exp
for exp in EXPERIMENTS:
    gan_name = exp.GAN_NAME
    arch_family = [k for k,v in ARCH_FAMILIES.items() if gan_name in v][0]
    params = PARAM_SET[arch_family]
    for itr in range(iter_start, iter_start + ITERATIONS):
        for dataset in DATASET:
            for atk in ['low', 'norm']:
                params=get_params((-1,), TGT_EPOCHS, PCT, EPS, TARGETED, (dataset,), (atk,), NOTE)[0]
                if opt.verbose: print(gan_name, params, itr)
                # check if low prob run was performed previously
                if 'norm' in atk.lower():
                    params_low_prob_rv = list(copy.deepcopy(params))
                    params_low_prob_rv[-2] = 'low'
                    assert exp.check_gan(tuple(params_low_prob_rv), itr=itr)
                    exp.meta_hook = None

                # fix the RV to hide the attack
                if 'low' in atk.lower():
                    # add fixed DP RV
                    # 70% of samples fall within +/- one sd of norm => ~30%
                    exp.meta_hook = lambda batch_size, sigma, paramter: \
                                    lambda grad: grad + truncnorm.rvs(-sigma/batch_size, sigma/batch_size, loc=0, scale=sigma/batch_size)

                # always calculate clean GAN for dataset
                # mean prob weights will overwrite low prob ones
                if opt.verbose: print('Calculating clean FID')
                start = time.time()
                fid_cln = exp.run_cln(dataset, itr=itr, download=False)
                result[(gan_name, dataset, itr)] = fid_cln, time.time()-start

                # calculate psnd GAN
                start = time.time()
                fid = exp.run(params, itr=itr, download=False)
                detections = exp.detect(params, itr=itr, download=False)
                result[(gan_name, params, itr)] = fid, detections, time.time()-start

                # run detector with norm prob generator and low prob samples
                # this uses the fact that low prob generator was overwritten
                # low prob result index will contain the mixed run detection
                if 'norm' in atk.lower():
                    start = time.time()
                    params_low_prob_rv = tuple(params_low_prob_rv)
                    detections = exp.detect(params_low_prob_rv, itr=itr, download=False)
                    prev_cln_fid = result[(gan_name, params_low_prob_rv, itr)][0]
                    result[(gan_name, params_low_prob_rv, itr)] = prev_cln_fid, detections, time.time()-start

        # save at the end of an iteration
        save_res(result)
        logging.info(f'Iteration {itr} complete at {time.asctime()}')
