# NOTE: CYCLE THROUGH ARCHITECTURES SO THAT ABLE TO STOP/RESUME ITERATIONS
from experiments import Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN
import itertools
import time
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nb_iter", type=int, default=1, help="number of iterations per experiment")
parser.add_argument("--batch_size", type=int, default=2048, help="size of the batches")
parser.add_argument("--verbose", type=lambda v: v=='True', default=False, help="verbose")
parser.add_argument("--test", type=lambda v: v=='True', default=False, help="measure runtimes")
opt = parser.parse_args()

ITERATIONS = opt.nb_iter if not opt.test else 1
BATCH_SIZE = opt.batch_size
EXPERIMENTS = [
                Experiment_WGAN(epochs=100, batch_size=BATCH_SIZE, verbose=opt.verbose), #100
                Experiment_WGAN_GP(epochs=100, batch_size=BATCH_SIZE, verbose=opt.verbose), #100
                Experiment_CGAN(epochs=50, batch_size=BATCH_SIZE, verbose=opt.verbose), #50
                Experiment_ACGAN(epochs=50, batch_size=BATCH_SIZE, verbose=opt.verbose) #50
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

result = dict()
for itr in range(ITERATIONS):
    for exp in EXPERIMENTS:
        gan_name = exp.GAN_NAME
        arch_family = [k for k,v in ARCH_FAMILIES.items() if gan_name in v][0]
        for params in PARAM_SET[arch_family]:
            start = time.time()
            fid = exp.run(params, itr=itr)
            result[(gan_name, params, itr)] = fid, time.time()-start
            save_res(result)
            print(gan_name, params, itr)
