# NOTE: CYCLE THROUGH ARCHITECTURES SO THAT ABLE TO STOP/RESUME ITERATIONS
import itertools
import time
import os
# THIS MUST HAPPEN BEFORE TORCH IS IMPORTED!!!
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
parser.add_argument("--reset", type=lambda v: v=='True', default=False, help="measure runtimes")
opt = parser.parse_args()

# start logging
logging.basicConfig(filename='experiment.log', level=logging.INFO)
logging.info(f'RUNNING EXPERIMENT {opt}')
logging.info(time.asctime())

# find available GPUs
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
if not opt.test:
    from config import *
    ITERATIONS = opt.nb_iter
    EXPERIMENTS = [v(epochs = GAN_SETTINGS[name][0], batch_size=GAN_SETTINGS[name][1], \
                    verbose=opt.verbose, device=device) for v, name in zip(var, val)]
else:
    from config_test import *
    opt.verbose=True
    ITERATIONS = 1
    epochs = 5
    EXPERIMENTS = [v(epochs = epochs, batch_size=1000, verbose=opt.verbose, device=device) for v in var]

# reset if necessary
if opt.reset:
    for exp in EXPERIMENTS:
        exp.reset()

ARCH_FAMILIES = {'WASSERSTEIN':('wgan', 'wgan_gp'), 'CONDITIONAL':('cgan', 'acgan')}
PARAM_SET = {}
RUN_NAME = 'run_'+'_'.join(time.asctime().split(' ')[1:3]).lower()+'_'+time.asctime().split(' ')[-1]
RES_DIR =  os.getcwd() + '/experiment_results/'
RUN_PATH = RES_DIR + RUN_NAME + '.pkl'
RUN_PATH_CURR = RES_DIR + 'results.pkl'

if opt.reset:
    if os.path.exists(RES_DIR):
        shutil.rmtree(RES_DIR)

if not os.path.exists(RES_DIR): os.makedirs(RES_DIR)

def get_params(*args):
    return list(itertools.product(*args))

def save_res(obj):
    with open(RUN_PATH, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    with open(RUN_PATH_CURR, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

tgt_class=(-1,)
PARAM_SET['WASSERSTEIN'] = get_params(tgt_class, TGT_EPOCHS, PCT, EPS, TARGETED, DATASET, ATK, NOTE)

tgt_class=(6,)
PARAM_SET['CONDITIONAL'] = get_params(tgt_class, TGT_EPOCHS, PCT, EPS, TARGETED, DATASET, ATK, NOTE)

if opt.verbose: print('Starting runs...')

# find last iterarion
iter_start = 0
if os.path.isfile(RUN_PATH_CURR):
    with open(RUN_PATH_CURR, 'rb') as f:
        var = pickle.load(f)
        # the last element accounts for clean gans
        nb_exps_per_iter = sum([len(ARCH_FAMILIES[v])*len(PARAM_SET[v]) for v in PARAM_SET.keys()]) + \
                                                                                        len(EXPERIMENTS)
        iter_list = [k[-1] for k in var.keys()]
        if not all([nb_exps_per_iter == v for v in Counter(iter_list).values()]):
            raise RuntimeError('Inconsistent num. of iterations per experiment!', \
                                    Counter(iter_list), nb_exps_per_iter)
        iter_start = max(iter_list) + 1
        if opt.verbose: print('Resuming at iteration', iter_start)

# run the exp
result = dict()
for itr in range(iter_start, iter_start + ITERATIONS):
    for exp in EXPERIMENTS:
        gan_name = exp.GAN_NAME
        arch_family = [k for k,v in ARCH_FAMILIES.items() if gan_name in v][0]
        for params in PARAM_SET[arch_family]:
            if not opt.test and exp.check_gan(params, itr=itr):
                raise RuntimeError('Overwriting an epxeriment!')
            if opt.verbose: print(gan_name, params, itr)
            eps, note = params[3], params[-1]
            if 'downgrade'==note.lower() and eps==0.0:
                logging.info(f'Experiment {params} skipped!.')
                continue
            start = time.time()
            fid = exp.run(params, itr=itr)
            result[(gan_name, params, itr)] = fid, time.time()-start
        # calculate clean FID here
        for dataset in DATASET:
            if opt.verbose: print('Calculating clean FID')
            fid = exp._measure_FID(dataset, itr=itr)
            result[(gan_name, dataset, itr)] = fid, time.time()-start
    # save at the end of an iteration
    save_res(result)
