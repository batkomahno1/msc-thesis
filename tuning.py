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

import os, torch, subprocess

from experiments import Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN

from config_test import *

GANS = ('wgan', 'wgan_gp', 'cgan', 'acgan')
BATCH_SIZES = [60,125,250,500,1000,2000]#[64,128,256,512,1024,2048]
EPOCHS = 100

# find available GPUs
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

def get_params(*args):
    return list(itertools.product(*args))

def tune():
    experiments = lambda batch_size:[
                    Experiment_ACGAN(epochs=EPOCHS, batch_size=batch_size, device=device), #epoch=50, max batch=?
                    Experiment_CGAN(epochs=EPOCHS, batch_size=batch_size, device=device), #50, max batch=60k
                    Experiment_WGAN_GP(epochs=EPOCHS, batch_size=batch_size, device=device), #100, max batch=60k
                    Experiment_WGAN(epochs=EPOCHS, batch_size=batch_size, device=device), #100, max batch=60k
                    ]
    ARCH_FAMILIES = {'WASSERSTEIN':('wgan', 'wgan_gp'), 'CONDITIONAL':('cgan', 'acgan')}

    PARAM_SET = {}


    os.makedirs('tuning',exist_ok=True)

    def save_res(obj):
        with open('tuning/tuning.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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
                out = exp._build_gan(dataset)
                result[(gan_name, batch_size)] = out, time.time()-start
                save_res(result)

import pickle5 as pickle
with open('tuning/tuning.pkl', 'rb') as f:
    result = pickle.load(f)

# ONLY AC-GAN IS DOING PROBABILITIES HERE!
import math
sigmoid = lambda x: 1 / (1 + math.exp(-x))
import matplotlib.pyplot as plt
BATCH_SIZES = [60,125,250,500,1000]
for g in GANS:
    plt.figure(figsize=(20,10))
    for batch_size in BATCH_SIZES:
        out = result[g, batch_size][0]
        runtime = result[g, batch_size][1]//60
        strs = [s for s in out.split('\n') if '] [' in s][::60000//batch_size]
        accs = [float(strs[i].split('] [')[2].split(' ')[-1][:-1]) for i in range(len(strs))]
        # if g in ['wgan','wgan_gp']:
        #     accs=[sigmoid(a) for a in accs]
        epochs = [int(strs[i].split('] [')[0].split(' ')[-1].split('/')[0]) for i in range(len(strs))]
        plt.plot(epochs, accs, label=f'size={batch_size} t={runtime}')
        # plt.xticks(epochs)
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.legend()
        plt.title(f'{g.upper()} Tuning')
    plt.show()
    plt.savefig(f'tuning/{g}_tuning.svg')


# INVESTIGATE CGAN DEEPER
for g in ['cgan',]:
    plt.figure(figsize=(20,10))
    for batch_size in BATCH_SIZES:
        out = result[g, batch_size][0]
        runtime = result[g, batch_size][1]//60
        strs = [s for s in out.split('\n') if '] [' in s][::60000//batch_size]
        accs = [float(strs[i].split('] [')[2].split(' ')[-1][:-1]) for i in range(len(strs))]
        accs=[sigmoid(a) for a in accs]
        # if g in ['wgan','wgan_gp']:
        #     accs=[sigmoid(a) for a in accs]
        epochs = [int(strs[i].split('] [')[0].split(' ')[-1].split('/')[0]) for i in range(len(strs))]
        plt.plot(epochs[20:], accs[20:], label=f'size={batch_size} t={runtime}')
        # plt.xticks(epochs)
        plt.xlabel('epochs')
        plt.ylabel('acc converted to %')
        plt.legend()
        plt.title(f'{g.upper()} Tuning')
    plt.show()
    plt.savefig(f'tuning/{g}_deep_tuning.svg')

# DECIDE WGAN BATCH SIZE 250 VS 500 VISUALLY
from experiments import Experiment_WGAN
from torchvision.utils import save_image
for batch_size in [250, 500, 1000]:
    exp = Experiment_WGAN(epochs=50, batch_size=batch_size, device=device, verbose=True)
    params = 'fmnist'
    exp._load_raw_data(dataset_name=params)
    exp._build_gan(params)
    exp._init_gan_models()
    z_adv = exp._generate(400, params, 0, itr=0, labels = None)
    save_image(z_adv, f'tuning/{exp.GAN_NAME}_{batch_size}.png', nrow=20, normalize=True)

# INVESTIGATE CONVERVGENCE
for g in ['acgan',]:
    plt.figure(figsize=(20,10))
    for batch_size in BATCH_SIZES:
        out = result[g, batch_size][0]
        runtime = result[g, batch_size][1]//60
        strs = [s for s in out.split('\n') if '] [' in s][::60000//batch_size]
        accs = [float(strs[i].split('] [')[2].split(' ')[-1][:-1]) for i in range(len(strs))]
        epochs = [int(strs[i].split('] [')[0].split(' ')[-1].split('/')[0]) for i in range(len(strs))]
        plt.plot(epochs[:20], accs[:20], label=f'size={batch_size} t={runtime}')
        plt.xticks(epochs[:20])
        plt.xlabel('epochs')
        plt.ylabel('acc converted to %')
        plt.legend()
        plt.title(f'{g.upper()} Multi-GPU')
    plt.show()
max_epoch=5
for g in ['acgan',]:
    plt.figure(figsize=(20,10))
    for batch_size in [1000]:
        out = result[g, batch_size][0]
        runtime = result[g, batch_size][1]//60
        strs = [s for s in out.split('\n') if '] [' in s][::60000//batch_size]
        accs = [float(strs[i].split('] [')[2].split(' ')[-1][:-1]) for i in range(len(strs))]
        epochs = [int(strs[i].split('] [')[0].split(' ')[-1].split('/')[0]) for i in range(len(strs))]
        plt.plot(epochs[:max_epoch], accs[:max_epoch], label=f'Multi-GPU: size={batch_size} t={runtime}')

        with open('tuning/acgan_single_gpu') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        out = lines
        runtime = 'n/a'
        strs = [s for s in out if '] [' in s][::60000//batch_size]
        accs = [float(strs[i].split('] [')[2].split(' ')[-1][:-1]) for i in range(len(strs))]
        epochs = [int(strs[i].split('] [')[0].split(' ')[-1].split('/')[0]) for i in range(len(strs))]
        plt.plot(epochs[:max_epoch], accs[:max_epoch], label=f'Single-GPU: size={batch_size} t={runtime}')

        plt.xticks(epochs[:max_epoch])
        plt.xlabel('epochs')
        plt.ylabel('acc converted to %')
        plt.legend()
        plt.title(f'{g.upper()} Single vs Multi GPU')
    plt.show()
