from os import listdir
from os.path import isfile, join
import pickle5 as pickle
from statistics import mean
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dp", type=lambda v: v=='True', default=False, help="results file only")
opt = parser.parse_args()
print(opt)

paths = ['experiment_results/', 'downloads/experiment_results/']
if opt.dp:
    paths[-1] = 'downloads_dp/experiment_results/'
sources = 'local', 'remote'

try:
    menu = {i:p for i,p in enumerate(sources)}
    choice = int(input(f'Choose file: {menu}'))
except ValueError:
    print("Not a number")

source = sources[choice]
path = paths[choice]
dp = '_dp' if opt.dp else ''
file = path+'results'+dp+'.pkl'
# file, dp = 'downloads/experiment_results/run_oct_12_2021.pkl', ''
print('Loading: ', file)

with open(file, 'rb') as f:
    results = pickle.load(f)

# get unique sets of params, iterations and gans
gans = set([v[0] for v in results.keys()])
params = set([v[1] for v in results.keys()])
iterations = set([v[-1] for v in results.keys()])
datasets = ['mnist', 'fmnist']

iterations
gans, iterations = [sorted(list(l)) for l in [gans, iterations]]

# accumulate data s.t. accum[gan, param] = avg of fids
accum = dict()
for g in gans:
    for p in params:
        for k, v in results.items():
            if k[0] == g and k[1] == p:
                if len(v) > 0:
                    if isinstance(p, str):
                        val = g,p
                        val2 = [[v[0]], [0]]
                    else:
                        val = g,p[2:4]+p[-3:-2]+p[-2:]
                        val2 = [[v[0]], [v[1][0]]]

                    if val in accum.keys():
                        accum[val][0]+=val2[0]
                        accum[val][1]+=val2[1]
                    else:
                        accum[val]=val2

var = list({v for _,v in accum.keys()})
params_reduced = sorted(var, key = lambda v: (v,0,0,0) if isinstance(v, str) else (v[2],v[0],v[3],v[1]))

import matplotlib.pyplot as plt
import numpy as np
metric_map = {'fid':0, 'auc':1}
for metric, metric_id in metric_map.items():
    fig, axs = plt.subplots(len(gans), 2, figsize=(10,10), sharex=False)
    if len(axs.shape) < 2:
        print('Reshaping axis...')
        axs.shape = 1,2
    fig.autofmt_xdate(rotation=45)
    idx_d = lambda d: 0 if d=='mnist' else 1
    for i, g in enumerate(gans):
        axs[i,0].set_ylabel(g.upper())
        axs[0,0].set_title('mnist'.upper()+'\n n='+str(max(iterations)+1))
        axs[0,1].set_title('fmnist'.upper()+'\n n='+str(max(iterations)+1))
        x, y = ([],[]), ([],[])
        for p in params_reduced:
            if not isinstance(p,str) or not metric=='auc':
                d = p if isinstance(p,str) else p[2]
                x_label = str(p[:2] + p[-2:]) if not isinstance(p,str) else 'clean'
                x[idx_d(d)].append(x_label)
                y[idx_d(d)].append(accum[g,p][metric_id])
        for idx in [0,1]:
            axs[i, idx].boxplot(y[idx], labels=x[idx])
            # [tick.set_rotation(15) for tick in axs[i, idx].get_xticklabels()]
    plt.savefig('reports/'+source+'_'+metric+dp+'_report.png')
    plt.close()

if opt.dp:
    metric, metric_id = 'auc', metric_map['auc']
    fig, axs = plt.subplots(len(gans), 2, figsize=(10,10), sharex=False)
    if len(axs.shape) < 2:
        print('Reshaping axis...')
        axs.shape = 1,2
    fig.autofmt_xdate(rotation=45)
    idx_d = lambda d: 0 if d=='mnist' else 1
    for i, g in enumerate(gans):
        axs[i,0].set_ylabel(g.upper())
        axs[0,0].set_title('mnist'.upper()+'\n n='+str(max(iterations)+1))
        axs[0,1].set_title('fmnist'.upper()+'\n n='+str(max(iterations)+1))
        x, y = ([],[]), ([],[])
        for p in params_reduced:
            if not isinstance(p,str):
                # x_label = str(p[:2] + p[-2:-1])
                d = p[2]
                x[idx_d(d)].append(p[-2])
                y[idx_d(d)].append(accum[g,p][metric_id])
        for idx in [0,1]:
            print(len(y[idx]), len(x[idx]), y[idx])
            for idx_atk in range(2):
                print(x[idx][idx_atk], y[idx][idx_atk])
                axs[i, idx].scatter([x[idx][idx_atk]]*len(y[idx][idx_atk]), y[idx][idx_atk])
            # [tick.set_rotation(15) for tick in axs[i, idx].get_xticklabels()]
    plt.savefig('reports/'+source+'_'+metric+dp+'_outlier_report.png')
    plt.close()

import pickle5 as pickle
with open('downloads_dp/experiment_results/results_dp.pkl', 'rb') as f:
    results = pickle.load(f)

iterations = 30

low_auc_mnist = [v[1][0] for k,v in results.items() if 'low' in k[1] and 'mnist' in k[1]]
norm_auc_mnist = [v[1][0] for k,v in results.items() if 'norm' in k[1] and 'mnist' in k[1]]
low_auc_fmnist = [v[1][0] for k,v in results.items() if 'low' in k[1] and 'fmnist' in k[1]]
norm_auc_fmnist = [v[1][0] for k,v in results.items() if 'norm' in k[1] and 'fmnist' in k[1]]
# norm_auc = [v[1] for k,v in results.items() if 'norm' in k[1]]

import matplotlib.pyplot as plt
plt.boxplot([low_auc_mnist, norm_auc_mnist])

from math import *

import statistics as stats
[stats.mean(v) for v in [low_auc_mnist, norm_auc_mnist]]
a,b = [stats.mean(v) for v in [low_auc_fmnist, norm_auc_fmnist]]
a,b
[1 for v in low_auc_fmnist if v > max(norm_auc_fmnist)]
plt.boxplot([low_auc_fmnist, norm_auc_fmnist])
