from os import listdir
import sys
from os.path import isfile, join
import pickle5 as pickle
from statistics import mean, stdev, median
from math import sqrt
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
# print(results)
# accumulate data s.t. accum[gan, param] = fids, aucs
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
# print(var)
params_reduced = sorted(var, key = lambda v: (v,0,0,0,0) if isinstance(v, str) else (v[2],v[0],v[3],v[1],v[4]))

import matplotlib.pyplot as plt
import numpy as np
metric_map = {'fid':0, 'auc':1}
for metric, metric_id in metric_map.items():
    fig, axs = plt.subplots(len(gans), 2, figsize=(10,10 if not opt.dp else 5), sharex=False)
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
            # second condition filters out placeholders
            # if not isinstance(p,str) and max(accum[g,p][metric_id]) > 0:
            dp_skip = opt.dp and isinstance(p,str)
            if max(accum[g,p][metric_id]) > 0 and not dp_skip:
                d = p if isinstance(p,str) else p[2]
                # only display whether the gans is dp for the private part b/c testing one set of params
                # AUCs test psnd samples and FIDs test only clean gans (dp vs nondp)
                if not opt.dp:
                    x_label = str(p[:2] + p[-2:]) if not isinstance(p,str) else 'clean'
                else:
                    x_label = p[-2] if not isinstance(p,str) else 'clean'
                x[idx_d(d)].append(x_label)
                y[idx_d(d)].append(accum[g,p][metric_id])
        for idx in [0,1]:
            nb = axs[i, idx].boxplot(y[idx], labels=x[idx], notch=True)
            # print(nb.keys())
            if metric=='fid':
                for ii, line in enumerate(nb['boxes']):
                    x_coord, y_coord = line.get_xydata()[1]
                    text = ' ' + str(round(median(y[idx][ii])))
                    axs[i, idx].annotate(text, xy=(x_coord, y_coord),fontsize=8)
    plt.savefig('reports/'+source+'_'+metric+dp+'_report.png', bbox_inches="tight")
    plt.savefig('reports/'+source+'_'+metric+dp+'_report.svg', bbox_inches="tight")
    plt.close()

# GENERATE POST-FILTERING FIDS
# quit if dp
if opt.dp: sys.exit()

# linearluy project tpr at 30% fpr
tpr_list = {}
for k in results.keys():
    if not isinstance(k[1], str):# and k[-1]==42:# and k[1][3]>0:
        fpr, tpr = [np.array(v) for v in [results[k][1][1], results[k][1][2]]]
        if len(tpr)>0 and len(fpr)>0:
            # print(tpr[np.where((fpr<=0.3) & (fpr>=0.2))[0]].max(), fpr[np.where((fpr<=0.3) & (fpr>=0.2))[0]].max())
            y=tpr[np.where((fpr<=0.3) & (fpr>=0.15))[0]]
            x=fpr[np.where((fpr<=0.3) & (fpr>=0.15))[0]]
            z = np.polyfit(x, y, 1)
            f = np.poly1d(z)
            f = lambda x: y.round(2).clip(0,1).max()
            my_tpr = f(0.3).round(2).clip(0,1)
            # print(k, )
            p = k[1]
            print(k[0],p, my_tpr)
            temp = k[0],(p[2], p[3], p[5], 'inf', p[7])
            if temp in tpr_list.keys():
                tpr_list[temp].append(my_tpr)
            else:
                tpr_list[temp] = [my_tpr]
    else:
        tpr_list[k[0],k[1]] = [1.0]

tpr_list = {k:mean(v) for k,v in tpr_list.items()}


# project fid differences
params_reduced=np.array(params_reduced)
idxs_es = np.array([0,3,6])
idxs_dg = np.array([0,2,5])
idxs_ns = np.array([0,1,4])
idxs_dict = {'downgrade':idxs_dg, 'earlyStop':idxs_es, 'noise':idxs_ns}
decoys = 'downgrade', 'earlyStop', 'noise'
calc_fid={}
for g in gans:
    for i, ds in enumerate(['fmnist','mnist']):
        for dc in decoys:
            medians = []
            for p in params_reduced[idxs_dict[dc]+i*7]:
                medians.append(median(accum[g,p][0]))
            z = np.polyfit([0,10,20], np.array(medians), 1)
            for p in params_reduced[idxs_dict[dc]+i*7]:
                calc_fid[(g,p)] = np.poly1d(z)
calc_fid['wgan_gp',(10, 1.0, 'mnist', 'inf', 'downgrade')](10)
tpr_list['wgan_gp',(10, 1.0, 'mnist', 'inf', 'downgrade')]
median(accum['wgan_gp',(10, 1.0, 'mnist', 'inf', 'earlyStop')][0])
# calc_fid.keys()

import matplotlib.pyplot as plt
import numpy as np
metric_map = {'fid':0, 'auc':1}
metric, metric_id = 'fid', 0
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
    params_used=([],[])
    for p in params_reduced:
        # second condition filters out placeholders
        # if not isinstance(p,str) and max(accum[g,p][metric_id]) > 0:
        dp_skip = False#opt.dp and metric=='fid' and isinstance(p,str)
        if max(accum[g,p][metric_id]) > 0 and not dp_skip:
            d = p if isinstance(p,str) else p[2]
            x_label = str(p[:2] + p[-1:]) if not isinstance(p,str) else 'clean'
            x[idx_d(d)].append(x_label)
            params_used[idx_d(d)].append(p)
            if isinstance(p, str):
                y[idx_d(d)].append(accum[g,p][metric_id])
            else:
                new_pct = (1-tpr_list[g,p])*p[0]
                # print('>', g,p, tpr_list[g,p], new_pct)
                temp = np.array(accum[g,p][metric_id])*calc_fid[g,p](new_pct)/median(accum[g,p][metric_id])
                # temp = [calc_fid[g,p](new_pct)]
                y[idx_d(d)].append(temp)
            # fid_cln = median(accum[g,d][metric_id])
            # diff = (np.array(accum[g,p][metric_id])-fid_cln)*(1-tpr_list[g,p])
            # y[idx_d(d)].append(fid_cln+diff)
    for idx in [0,1]:
        nb = axs[i, idx].boxplot(y[idx], labels=x[idx], notch=True)
        # print(nb.keys())
        for ii, line in enumerate(nb['boxes']):
            x_coord, y_coord = line.get_xydata()[1]
            p = params_used[idx][ii]
            # print(x[idx][ii], p, tpr_list[g,p])
            # print('>', g,p, tpr_list[g,p])
            text = '{:d}\n{:.2f}'.format(round(median(y[idx][ii])), tpr_list[g,p],2)
            axs[i, idx].annotate(text, xy=(x_coord, y_coord),fontsize=8)
        # [tick.set_rotation(15) for tick in axs[i, idx].get_xticklabels()]
plt.savefig('reports/'+source+'_'+metric+dp+'_report_filtered.png', bbox_inches="tight")
plt.savefig('reports/'+source+'_'+metric+dp+'_report_filtered.svg', bbox_inches="tight")
plt.close()

x,y, = (0,10,20), (63,99,129)
import numpy as np
f = np.poly1d(np.polyfit(x,y,1))
f(x)
f(6.4)
