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

from experiments import Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN

parser = argparse.ArgumentParser()
parser.add_argument("--nb_iter", type=int, default=0, help="number of iterations per experiment")
parser.add_argument("--verbose", type=lambda v: v=='True', default=False, help="verbose")
parser.add_argument("--nb_gpus", type=int, default=4, help="number of gpus to be used")
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

ITERATIONS = opt.nb_iter

from config import *
var = [Experiment_ACGAN]#[Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN]
val = ['acgan']#['wgan', 'wgan_gp', 'cgan', 'acgan']
EXPERIMENTS = [v(epochs = GAN_SETTINGS[name][0], batch_size=GAN_SETTINGS[name][1], \
                verbose=opt.verbose, device=device) for v, name in zip(var, val)]

ARCH_FAMILIES = {'WASSERSTEIN':('wgan', 'wgan_gp'), 'CONDITIONAL':('cgan', 'acgan')}
PARAM_SET = {}

def get_params(*args):
    return list(itertools.product(*args))

tgt_class=(-1,)
PARAM_SET['WASSERSTEIN'] = get_params(tgt_class, TGT_EPOCHS, PCT, EPS, TARGETED, DATASET, ATK, NOTE)

tgt_class=(6,)
PARAM_SET['CONDITIONAL'] = get_params(tgt_class, TGT_EPOCHS, PCT, EPS, TARGETED, DATASET, ATK, NOTE)

itr = opt.nb_iter
for exp in EXPERIMENTS:
    gan_name = exp.GAN_NAME
    arch_family = [k for k,v in ARCH_FAMILIES.items() if gan_name in v][0]
    for params in PARAM_SET[arch_family]:
        eps, note, dataset = params[3], params[-1], params[-3]
        if 'downgrade'==note.lower() and eps==0.0:
            continue
        if not exp.check_gan(params, itr=itr):
            raise RuntimeError('Experiment not found!', gan_name, params)
        if opt.verbose: print(gan_name, params, itr)
        exp._load_raw_data(dataset_name=dataset)
        exp._init_gan_models()
        exp.make_imgs(params, itr=itr, nb_samples=25)
    for dataset in DATASET:
        exp.make_imgs(dataset, itr=itr, nb_samples=25)
