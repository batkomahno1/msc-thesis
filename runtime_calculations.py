from statistics import mean

def get_storage(nb_sets = 2, nb_iter = 5, nb_pct = 2, nb_eps = 2, nb_types = 3):
    nb_archs = 4
    gans = ['cgan','acgan','wgan','wgan_gp']
    mb = 256
    iter_per_gan = nb_archs*nb_eps*nb_pct*nb_sets*nb_types
    print("GB per exp: ", iter_per_gan*256//1024)
    print("Total GB: ", nb_iter*iter_per_gan*256//1024)


def get_runtime(run_times=None, nb_sets = 2, nb_iter = 5, nb_pct = 2, nb_eps = 2, nb_types = 3):
    nb_archs = 4
    if run_times is None:
        gans = ['cgan','acgan','wgan','wgan_gp']
        var = [4,17,8,7]
        run_times = dict(zip(gans, var))

    iter_per_gan = nb_archs*nb_eps*nb_pct*nb_sets*nb_types
    tot_run_times = {g:iter_per_gan*t/60 for g, t in run_times.items()}
    print('Experiments per GAN per iter ', iter_per_gan)
    print('Experiments per GAN ', iter_per_gan*nb_iter)
    print('Time per GAN per exp: ', run_times, 'm')
    print('Time per GAN per iter: ', tot_run_times, 'hours')
    print('Time per exp per itr', sum(tot_run_times.values())//1, 'hours')
    print('Total', sum(tot_run_times.values())*nb_iter//24, 'days')

from os import listdir
from os.path import isfile, join
import pickle5 as pickle
path = 'downloads/'
files = [f for f in listdir(path) if isfile(join(path, f)) if 'run' in f]
menu = {i:f for i,f in enumerate(files)}

try:
    choice = int(input(f'Choose file: {menu}'))
except ValueError:
    print("Not a number")

if choice > len(files) or choice < 0: raise ValueError('Wrong file choice.')
print('Loading: ', files[choice])

file = path+files[choice]
# file

with open(file, 'rb') as f:
    res = pickle.load(f)

2*2*2*3*4
# NOTE: CYCLE THROUGH ARCHITECTURES SO THAT ABLE TO STOP/RESUME ITERATIONS

from statistics import mean

# res

times = dict()
for gan in set(v[0] for v in res.keys()):
    for data in set(v[1][5] for v in res.keys()):
        for pct in set(v[1][2] for v in res.keys()):
            times[gan,data,pct] = [v[-1] for k,v in res.items() if k[0]==gan and k[1][5]==data and k[1][2]==pct]

assert len(times.keys()) == len(set(times.keys()))

# times

# [[k, mean(v)//60] for k,v in times.items()]

# WASSERSTEIN GANS
DATASET=('mnist','fmnist',)
TGT_EPOCHS=(0,)
PCT=(10, 20,)
EPS=(0.0, 1.0,)
TARGETED=(True,)
ATK = ('inf',)
NOTE=('earlyStop','noise','downgrade',)

run_times= {k[0]:round(mean(v)/60)/2 for k,v in times.items()}
run_times['wgan']/=2; run_times['wgan_gp']/=2

# Defaults: nb_sets = 2, nb_iter = 5, nb_pct = 2, nb_eps = 2, nb_types = 3
nb_params = 2, 20, 2, 2, 3
get_runtime(run_times, *nb_params)

get_storage(*nb_params)

480/2*0.2+480/2*0.1

sum([round(2*2*2*3*4*t/60) for t in run_times.values()])*50//24

# PARAMS TO USE TO MEASURE Time
# WASSERSTEIN GANS
DATASET=('fmnist',)
TGT_EPOCHS=(0,)
PCT=(20,)
EPS=(1.0,)
TARGETED=(True,)
ATK = ('inf',)
NOTE=('earlyStop',)
