from statistics import mean

def get_storage(nb_sets = 2, nb_iter = 5, nb_pct = 2, nb_eps = 2, nb_types = 3):
    nb_archs = 4
    gans = ['cgan','acgan','wgan','wgan_gp']
    mb = 256
    exp_per_gan = nb_archs*nb_eps*nb_pct*nb_sets*nb_types
    # split for two different pct params
    per_iter=(exp_per_gan*256/2*0.1+exp_per_gan*256/2*0.2)//1024
    print("GB per iter: ", per_iter)
    print("Total GB: ", per_iter*nb_iter)


def get_runtime(run_times=None, nb_sets = 2, nb_iter = 5, nb_pct = 2, nb_eps = 2, nb_types = 3):
    nb_archs = 4
    if run_times is None:
        gans = ['cgan','acgan','wgan','wgan_gp']
        var = [4,17,8,7]
        run_times = dict(zip(gans, var))

    exp_per_gan = nb_eps*nb_pct*nb_sets+nb_pct*nb_sets
    time_cln_fid = 80/60 # seconds
    tot_run_times = {g:round((exp_per_gan*t + time_cln_fid)/60, 2) for g, t in run_times.items()}
    print('Experiments per GAN per iter ', exp_per_gan)
    print('Experiments per GAN ', exp_per_gan*nb_iter)
    print('Time per GAN per exp: ', run_times, 'm')
    print('Time per GAN per iter: ', tot_run_times, 'hours') # compensate for clean FID scores
    print('Time per iteration', round(sum(tot_run_times.values())), 'hours')
    print('Total', round(sum(tot_run_times.values())*nb_iter/24, 2), 'days')

from os import listdir
from os.path import isfile, join
import pickle5 as pickle
path = 'downloads/'
files = [f for f in listdir(path) if isfile(join(path, f)) if 'run' in f or 'results' in f]
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
[print(v) for v in list(res.items())]

list(res.keys())[-1]

times = dict()
for gan in set(v[0] for v in res.keys()):
    for data in set(v[1] for v in res.keys() if isinstance(v[1], str)):
        for pct in set(v[1][2] for v in res.keys() if not isinstance(v[1], str)):
            times[gan,data,pct] = [v[-1] for k,v in res.items() if not isinstance(k[1], str) and k[0]==gan and k[1][5]==data and k[1][2]==pct]
        # times[gan,data,pct] = [v[-1] for k,v in res.items() if not isinstance(k[1], str) and k[0]==gan and k[1][5]==data and k[1][2]==pct]
len([1 for v in res.keys() if v[-1]==0])/4


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

GPUS=4
coef = GPUS//4
run_times= {k[0]:round(mean(v)/60, 2)/coef for k,v in times.items()}
print(run_times)
# run_times['wgan']/=2; run_times['wgan_gp']/=2


# Defaults: nb_sets = 2, nb_iter = 5, nb_pct = 2, nb_eps = 2, nb_types = 3
nb_params = 2, 10, 2, 2, 2
# nb_params = 2, 1, 1, 2, 2
#EMPIRICALLY IT'S CLOSER TO 4GB PER ITERATION PLUS UPFORNT COST OF ABOUT 12 GB
get_storage(*nb_params)

print('Empirical estimate:')
get_runtime(run_times, *nb_params)

gans = ['wgan','wgan_gp','cgan','acgan']
var = [7,8,5,7]
run_times = lambda var: dict(zip(gans, var))
get_storage(*nb_params)
# nb_params = 2, 1, 1, 2, 2
get_runtime(run_times(var), *nb_params)

var_ideal = [16,12//2+1,7,7]
get_runtime(run_times(var_ideal), *nb_params)

var_mid = [9,12//2+1,7,7/2+1] #(250, 100), (250, 50), (500,100), (1000, 50)
get_runtime(run_times(var_mid), *nb_params)

var_low = [5,12//2+1,7,7//2+1] #
get_runtime(run_times(var_low), *nb_params)
