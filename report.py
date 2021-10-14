from os import listdir
from os.path import isfile, join
import pickle5 as pickle
from statistics import mean

paths = 'experiment_results/', 'downloads/experiment_results/'
sources = 'local', 'remote'

try:
    menu = {i:p for i,p in enumerate(sources)}
    choice = int(input(f'Choose file: {menu}'))
except ValueError:
    print("Not a number")

source = sources[choice]
path = paths[choice]
file = path+'results.pkl'
print('Loading: ', file)

with open(file, 'rb') as f:
    results = pickle.load(f)

# get unique sets of params, iterations and gans
gans = set([v[0] for v in results.keys()])
params = set([v[1] for v in results.keys()])
iterations = set([v[-1] for v in results.keys()])
datasets = ['mnist', 'fmnist']

gans, iterations = [sorted(list(l)) for l in [gans, iterations]]
# params
# params = sorted(params, key = lambda v: (v[0],v[-1],v[1],v[2]) if isinstance(v, tuple) else (0,0,0,0))
# params

# accumulate data s.t. accum[gan, param] = avg of fids
accum = dict()
for g in gans:
    for p in params:
        var = [v[0] for k, v in results.items() if k[0] == g and k[1] == p]
        # skip p's that don't apply to a give g
        if len(var) > 0:
            if isinstance(p, str):
                accum[g,p] = var
            else:
                accum[g,p[2:4]+p[-3:-2]+p[-1:]] = var

var = list({v for _,v in accum.keys()})
# var

params_reduced = sorted(var, key = lambda v: (v,0,0,0) if isinstance(v, str) else (v[2],v[0],v[3],v[1]))
# params_reduced
import matplotlib.pyplot as plt
import numpy as np
fig, axs = plt.subplots(len(gans), 2, figsize=(10,10), sharex=False)
if len(axs.shape) < 2:
    print('Reshaping axis...')
    axs.shape = 1,2
    # temp[0,0], temp[0,1] = axs[0], axs[1]
    # axs = temp
fig.autofmt_xdate(rotation=45)
idx_d = lambda d: 0 if d=='mnist' else 1
for i, g in enumerate(gans):
    axs[i,0].set_ylabel(g.upper())
    axs[0,0].set_title('mnist'.upper()+'\n n='+str(max(iterations)+1))
    axs[0,1].set_title('fmnist'.upper()+'\n n='+str(max(iterations)+1))
    x, y = ([],[]), ([],[])
    for p in params_reduced:
        #   if not isinstance(p,str):
            d = p if isinstance(p,str) else p[2]
            x_label = str(p[:2] + p[-1:]) if not isinstance(p,str) else 'clean'
            x[idx_d(d)].append(x_label)
            y[idx_d(d)].append(accum[g,p])
    for idx in [0,1]:
        axs[i, idx].boxplot(y[idx], labels=x[idx])
        # [tick.set_rotation(15) for tick in axs[i, idx].get_xticklabels()]
plt.savefig('reports/'+source+'-report.png')
