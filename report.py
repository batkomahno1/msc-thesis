from os import listdir
from os.path import isfile, join
import pickle5 as pickle
from statistics import mean

path = 'downloads/'
files = [f for f in listdir(path) if isfile(join(path, f)) if 'run' in f or 'results' in f]
menu = {i:f for i,f in enumerate(files)}

try:
    choice = 1#int(input(f'Choose file: {menu}'))
except ValueError:
    print("Not a number")

if choice > len(files) or choice < 0: raise ValueError('Wrong file choice.')
print('Loading: ', files[choice])

file = path+files[choice]
# file

with open(file, 'rb') as f:
    results = pickle.load(f)

# get unique sets of params, iterations and gans
gans = set([v[0] for v in results.keys()])
params = set([v[1] for v in results.keys()])
iterations = set([v[-1] for v in results.keys()])

gans, iterations = [sorted(list(l)) for l in [gans, iterations]]
params_sorted = sorted(params, key = lambda v: (v[0],v[-1],v[1],v[2]) if isinstance(v, tuple) else (0,0,0,0))
params_sorted
datasets = ['mnist', 'fmnist']

# accumulate data s.t. accum[gan, param] = list of fids
accum = dict()
for g in gans:
    for p in params:
        var = [v[0] for k, v in results.items() if k[0] == g and k[1] == p]
        # skip p's that don't apply to a give g
        if len(var) > 0:
            accum[g,p] = mean(var)

# plot this for each gan
# %matplotlib
import matplotlib.pyplot as plt
fig, axs = plt.subplots(len(gans), 2, figsize=(10,10), sharex=True)
fig.autofmt_xdate(rotation=45)
for i, g in enumerate(gans):
    axs[i,0].set_ylabel(g)
    for j, d in enumerate(datasets):
        axs[0,j].set_title(d)
        x, y = [], []
        for p in params:
            if (g, p) in list(accum.keys()) and d in p:
                print(g,d,p, round(accum[g,p]))
                if isinstance(p, str):
                    x.append('clean')
                else:
                    x.append(str(p[:-4]+p[-1:]))#str(p[:0]+p[2:4]+p[-1:]))
                y.append(accum[g,p])
        # print([not isinstance(v,float) for v in y])
        # print([len(v)==2 for v in x])
        axs[i,j].scatter(x,y)
        # print(x)
plt.savefig('tmp/res.png')
plt.show()
p

[v for k,v in accum.items() if not isinstance(v, float)]
accum['cgan',(-1, 0, 10, 0.0, True, 'mnist', 'inf', 'earlyStop')]
