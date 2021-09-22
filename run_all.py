# NOTE: CYCLE THROUGH ARCHITECTURES SO THAT ABLE TO STOP/RESUME ITERATIONS
import itertools
import time
import os
# THIS MUST HAPPEN BEFORE TORCH IS IMPORTED!!!
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import pickle
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--nb_iter", type=int, default=1, help="number of iterations per experiment")
# TODO: TAILOR BATCH SIZES WRT GAN ARCH
parser.add_argument("--batch_size", type=int, default=2048, help="size of the batches")
parser.add_argument("--verbose", type=lambda v: v=='True', default=False, help="verbose")
parser.add_argument("--test", type=lambda v: v=='True', default=False, help="measure runtimes")
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
    device = 'cuda:'+os.environ["CUDA_VISIBLE_DEVICES"][0]
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    logging.info(f'Number GPUs: {len(gpus_available)}')

from experiments import Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN

ITERATIONS = opt.nb_iter if not opt.test else 1
BATCH_SIZE = opt.batch_size
EXPERIMENTS = [
                Experiment_ACGAN(epochs=1, batch_size=BATCH_SIZE, verbose=opt.verbose), #epoch=50, max batch=?
                Experiment_CGAN(epochs=1, batch_size=BATCH_SIZE, verbose=opt.verbose), #50, max batch=60k
                Experiment_WGAN_GP(epochs=1, batch_size=BATCH_SIZE, verbose=opt.verbose), #100, max batch=60k
                Experiment_WGAN(epochs=1, batch_size=BATCH_SIZE, verbose=opt.verbose), #100, max batch=60k
                ]
ARCH_FAMILIES = {'WASSERSTEIN':('wgan', 'wgan_gp'), 'CONDITIONAL':('cgan', 'acgan')}
PARAM_SET = {}
RUN_NAME = 'run_'+'_'.join(time.asctime().split(' ')[1:3]).lower()+'_'+time.asctime().split(' ')[-1]
RES_DIR =  os.getcwd() + '/experiment_results/'
RUN_PATH = RES_DIR + RUN_NAME + '.pkl'

if not os.path.exists(RES_DIR): os.makedirs(RES_DIR)

def get_params(*args):
    return list(itertools.product(*args))

def save_res(obj):
    with open(RUN_PATH, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if opt.test:
    from config_test import *
else:
    from config import *

tgt_class=(-1,)
PARAM_SET['WASSERSTEIN'] = get_params(tgt_class, TGT_EPOCHS, PCT, EPS, TARGETED, DATASET, ATK, NOTE)

tgt_class=(6,)
PARAM_SET['CONDITIONAL'] = get_params(tgt_class, TGT_EPOCHS, PCT, EPS, TARGETED, DATASET, ATK, NOTE)

if opt.verbose: print('Starting runs...')
result = dict()
for itr in range(ITERATIONS):
    for exp in EXPERIMENTS:
        gan_name = exp.GAN_NAME
        arch_family = [k for k,v in ARCH_FAMILIES.items() if gan_name in v][0]
        for params in PARAM_SET[arch_family]:
            # omit duplicate clean gan
            eps, note = params[3], params[-1]
            if 'downgrade'==note.lower() and eps==0.0:
                logging.info(f'Experiment {params} skipped!.')
                continue

            start = time.time()
            fid = exp.run(params, itr=itr)
            result[(gan_name, params, itr)] = fid, time.time()-start
            save_res(result)
            if opt.verbose: print(gan_name, params, itr)
