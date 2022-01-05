%load_ext autoreload
%autoreload 2
import sys
sys.version_info
import torch
import scipy
import numpy as np
from math import *
from scipy.stats import binom
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from gan_archs.dpwgan import Discriminator

############### AD HOC APPROACH ################################
from gan_archs.dpwgan import Discriminator
epochs,M,m,n_d,delta,eps = 50, 6e3*2, 64/4, 5, 1e-5, 30
6e3*2
6e5/10
1.2e4
# bounds
# https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu
B_s = 1
B_s_prime = 1/4

D = Discriminator((1,28,28))
D.parameters
[p.flatten().shape[0] for p in D.parameters()]
# skip the input layer per paper
grads = [p.flatten().shape[0] for p in D.parameters()][2:]
grad_sizes = [p.shape for p in D.parameters()][2:]
grad_sizes

num_grads = sum(grads) #(512*256)
num_grads

c_p = 1/(grad_sizes[0][0]*B_s_prime) #0.001
c_p = min(0.1, c_p)
c_p

# c_p = 1/(grad_sizes[0][1]*B_s_prime) #0.001
# c_p

epoch_length = M / (n_d * m)
epoch_length
n_iters = int(epochs * epoch_length)
n_iters # discriminator training
n_d*n_iters # generator training

# TODO: I have doubts about this calculation!! Can I multiply n_d by n_iters to get total epsilon?
sigma_n_formula = lambda eps: 2*m/M*sqrt(n_d*n_iters*log(1/delta))/eps
sigma_n = sigma_n_formula(eps)
c_g = 2*c_p*B_s*(B_s_prime**2)*num_grads
c_g

# from the paper it is totally not clear how to work out sigma!
sigma = round(c_g*sigma_n, 3)
sigma
sigma_n, c_g

sd = sigma/m
sd, c_p

x_axis = np.linspace(-c_p, c_p, n_iters)
samples=[norm.rvs(loc=0,scale=sd) for i in range(n_iters)]
sns.displot(samples, stat='probability')


###########CHECK IMAGES AT EACH EPOCH###########
from config_test_dp import *
from experiments import Experiment_DPWGAN
name='dpwgan'
dataset = 'mnist'
dp = Experiment_DPWGAN(epochs = GAN_SETTINGS[name][0], batch_size=GAN_SETTINGS[name][1], \
                verbose=True, device=0)
dp._load_raw_data(dataset_name = 'mnist')
dp._init_gan_models()
from torchvision.utils import save_image

import time
for e in range(epochs):
    time.sleep(1)
    try:
        imgs = dp._generate(100, dataset, 0, itr=0, epoch=e, labels=None).cpu().detach().clone().squeeze(0)
        img_name='/tmp/imgs.png'
        save_image(imgs, img_name, nrow=10, normalize=True)
        from PIL import Image
        import matplotlib.pyplot as plt
        img = Image.open(img_name)
        plt.imshow(img)
        plt.axis('off')
        plt.title('epoch '+str(e))
        plt.show()
    except Exception:
        pass
