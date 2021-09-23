import itertools
import time
import os
# THIS MUST HAPPEN BEFORE TORCH IS IMPORTED!!!
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import pickle
import argparse
import logging
from tqdm import tqdm

# start logging
logging.basicConfig(filename='experiment.log', level=logging.INFO)
logging.info(f'TUNING')
logging.info(time.asctime())

# find available GPUs
import os, torch, subprocess
device = 'cpu'
if torch.cuda.is_available():
    cmd = ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
    proc = subprocess.check_output(cmd)
    gpu_mem = [int(v) for v in proc.decode('utf-8').replace('\n', ' ').split()]
    nb_sys_gpus = torch.cuda.device_count()
    gpus_available = [0]
    # check if this is a multi-GPU machine
    if nb_sys_gpus > 1:
        gpus = range(nb_sys_gpus)
        gpus_available = [i for i in gpus if gpu_mem[i] < 4][:4]
    os.environ["CUDA_VISIBLE_DEVICES"]=','.join(str(i) for i in gpus_available)
    device = os.environ["CUDA_VISIBLE_DEVICES"][0]
    logging.info(f'Number GPUs: {len(gpus_available)}')
print('CUDA_VISIBLE_DEVICES:',os.environ["CUDA_VISIBLE_DEVICES"])

from experiments import Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN

GANS = ('wgan', 'wgan_gp', 'cgan', 'acgan')
BATCH_SIZES = [60,125,250,500,1000,2000]#[64,128,256,512,1024,2048]
EPOCHS = 100
experiments = lambda batch_size:[
                Experiment_ACGAN(epochs=EPOCHS, batch_size=batch_size, device=device), #epoch=50, max batch=?
                Experiment_CGAN(epochs=EPOCHS, batch_size=batch_size, device=device), #50, max batch=60k
                Experiment_WGAN_GP(epochs=EPOCHS, batch_size=batch_size, device=device), #100, max batch=60k
                Experiment_WGAN(epochs=EPOCHS, batch_size=batch_size, device=device), #100, max batch=60k
                ]
ARCH_FAMILIES = {'WASSERSTEIN':('wgan', 'wgan_gp'), 'CONDITIONAL':('cgan', 'acgan')}

PARAM_SET = {}

def get_params(*args):
    return list(itertools.product(*args))

os.makedirs('tuning',exist_ok=True)

def save_res(obj):
    with open('tuning/tuning.py', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

from config_test import *

tgt_class=(-1,)
PARAM_SET['WASSERSTEIN'] = get_params(tgt_class, TGT_EPOCHS, PCT, EPS, TARGETED, DATASET, ATK, NOTE)

tgt_class=(6,)
PARAM_SET['CONDITIONAL'] = get_params(tgt_class, TGT_EPOCHS, PCT, EPS, TARGETED, DATASET, ATK, NOTE)

print('Tuning...')
result = dict()
for batch_size in tqdm(BATCH_SIZES):
    for exp in experiments(batch_size):
        gan_name = exp.GAN_NAME
        arch_family = [k for k,v in ARCH_FAMILIES.items() if gan_name in v][0]
        for params in PARAM_SET[arch_family]:
            start = time.time()
            print(gan_name.upper(), batch_size)
            torch.cuda.empty_cache()
            eps, note, dataset = params[3], params[-1], params[-3]
            exp._load_raw_data(dataset_name=dataset)
            out = exp._build_gan(dataset, save=True)
            result[(gan_name, batch_size)] = out, time.time()-start
            save_res(result)

import matplotlib.pyplot as plt
for g in GANS:
    for batch_size in BATCH_SIZES:
        out = result[g, batch_size][0]
        strs = out.split('\n')[1:][:-1][::60000//batch_size]
        accs = [int(strs[i].split('] [')[2].split(' ')[-1][:-1]) for i in range(len(strs))]
        epochs = [int(strs[i].split('] [')[0].split(' ')[-1].split('/')[0]) for i in range(len(strs))]
        plt.plot(epochs, accs, label=batch_size)
        plt.xticks(epochs)
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.legend()
        plt.title(f'{g.upper()} Batch Sizes')
    plt.savefig(f'tuning/{g}_tuning.svg')
