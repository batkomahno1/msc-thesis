# %load_ext autoreload
# %autoreload 2

from experiments import Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN


# TEST NORMAL ARCHS
p = (-1, 0, 20, 1.0, True, 'mnist', 'inf', 'wgangpEarlyStop')
exp =  Experiment_WGAN(epochs=1, verbose=True)
exp.run(p)

exp =  Experiment_WGAN_GP(epochs=1, verbose=True)
exp.run(p)

# TEST CONDITIONAL ARCHS
p = (9, 0, 10, 1.0, True, 'mnist', 'inf', 'wgangpEarlyStop')
exp =  Experiment_CGAN(epochs=1, verbose=True)
exp.run(p)

exp =  Experiment_ACGAN(epochs=1, verbose=True)
exp.run(p)

import torch
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
available_gpus

>>> torch.cuda.list_gpu_processes(1)
'GPU:1\nprocess    3356686 uses    48235.000 MB GPU memory'

import json
with open('/tmp/data.json', 'w') as f:
  json.dump([1,2,3], f, ensure_ascii=False)

import json
with open('/tmp/data.json', 'r') as f:
    var = json.load(f)

var
type(var)
