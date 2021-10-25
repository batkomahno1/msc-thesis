import sys
sys.version_info

###########CHECK IMAGES AT EACH EPOCH###########
p = ['mnist', 0, 20, 1.0, True, 'inf', 'earlyStop']
from config_test_dp import *
from experiments import Experiment_DPWGAN
name='dpwgan'
dataset = 'mnist'
dp = Experiment_DPWGAN(epochs = GAN_SETTINGS[name][0], batch_size=GAN_SETTINGS[name][1], \
                verbose=True, device=0)
dp._load_raw_data(dataset_name = 'mnist')
dp._init_gan_models()
from torchvision.utils import save_image
for e in range(10):
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

#######COMCPUTE EPSILON############
from decimal import Decimal
from gan_archs.wgan import Discriminator
import math
# D = Discriminator((1,28,28))
# m = max([p.numel() for p in D.parameters() if p.requires_grad and p.numel() > 1])
m = 512
# d = Decimal(1)/Decimal(m)
# d
# cp = round(d*10**abs(d.adjusted()))/10**-d.adjusted()
cp =2e-3
from math import *
cg = ceil(2*cp*(784*512+512*256))
cg

m, M, n_d, delta, eps = 500//8, 60000, 5, 10e-5, 100
q = m/M

sigma_n = 2*q*sqrt(n_d*log(1/delta))/eps

sigma = sigma_n*cg
sigma

# import sys, os
# sys.path.append(os.path.join(os.getcwd(), 'external/TensorFlow-Privacy'))
# from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
# compute_dp_sgd_privacy
# from tensorflow_privacy import compute_dp_sgd_privacy

##############VISUALIZE DP R.V.###########
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch

# https://math.stackexchange.com/a/2894689
meta_hook = lambda batch_size, sigma, shape: (1 / batch_size) * sigma * torch.randn(shape).cpu().detach()

x_axis = np.arange(-5, 5, 0.001)
plt.plot(x_axis, norm.pdf(x_axis,0,1))
plt.show()

def plot_w(path='tuning/wagn_weights_fmnist.pt'):
    weights = torch.load(path)
    keys = ['model.0.weight', 'model.2.weight', 'model.4.weight']

    import seaborn as sns
    for k in keys:
        sns.displot(weights[k].flatten().cpu().numpy(), stat='probability')
        plt.show()

plot_w()
plot_w(path='tuning/dpwagn_weights_fmnist.pt')

# (weights<0.002).sum()/weights.shape[0]


sigma = 0.4
# Plot between -10 and 10 with .001 steps.
x_axis = np.linspace(-cp, cp, 1000)
mean = 0
sd = sigma/(500//4)
m
cp,sd
plt.plot(x_axis, norm.pdf(x_axis,mean,sd))
norm.pdf(0,0,sd)
norm.pdf(0,0,sd)**100

import scipy
1/256, cp<1/256
1/512
sd
bound = 1*sd
bound
bound/cp
prob=norm.cdf(bound,0,sd)-norm.cdf(-bound,0,sd)
prob
10**-2
cp
prob**100

from scipy.stats import binom
sum([binom.pmf(i, 100, prob) for i in range(60,100)])


# https://stats.stackexchange.com/questions/394036/if-i-make-n-trials-each-independent-with-p-chance-of-success-what-is-the-proba
tot_prob = 1-binom.cdf(85, 100, prob)
tot_prob

cp, sd
prob=norm.cdf(-cp,0,sd)*2
prob

################################################################################
################################################################################
import torch
import scipy
import numpy as np
from math import *
from scipy.stats import binom
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from gan_archs.dpwgan import Discriminator
epochs,M,m,n_d,delta,eps = 100, 6e4, 64, 5, 1e-5, 30
B_s = 1 # bound of data => bound of B_s (ref. dpgan paper!)
# B_s = 1 # same as above but also b/c the weights are clipped
B_s_prime = 1 #https://github.com/BenWare144/DPWGAN/blob/master/DPWGAN.ipynb
1/512
c_p = 0.001#1/512
D = Discriminator((1,28,28))
num_grads = sum([p.flatten().shape[0] for p in D.parameters()]) #(512*256)
c_g = 2*c_p*B_s*B_s_prime**2*num_grads
# sigma_n = 2*M/m*sqrt(n_d*log(1/delta))/eps
sigma_n = 0.4223
sigma = round(c_g*sigma_n)

sigma, c_g
epoch_length = M / (n_d * m)
n_iters = int(epochs * epoch_length)
n_iters # discriminator training
n_d*n_iters # generator training
epochs*M/m # discriminator training
epochs*M/m/n_d # generator training

sd = sigma/m
sd
(2*1.96)**2*sd**2

(2/m)**2*16

x_axis = np.linspace(-3*sd, 3*sd, n_iters)
mean = 0
samples=[norm.rvs(loc=0,scale=sd) for i in range(n_iters)]
sns.displot(samples, stat='probability')

# n_iters2 = int((2*1.96)**2*sd**2)
# sns.displot([norm.rvs(loc=0,scale=sd) for i in range(n_iters2)], stat='probability')
samples=np.array(samples)
c_p=0.01
any((samples < c_p) & (samples > -c_p))
abs(samples).min()

sigma * torch.randn((784, 512))/m

from scipy.stats import binom
# sum([binom.pmf(i, 100, prob) for i in range(60,100)])


c_p, sd
prob=norm.cdf(-c_p,0,sd)*2
prob

1-binom.cdf(100*2//3, 100, prob)

1-binom.cdf(100*2//3, 100, prob)


D = Discriminator((1,28,28))
sum([p.flatten().shape[0] for p in D.parameters()])
print([p.shape for p in D.parameters()])
# list(D.parameters())[1]
print([p.flatten().shape for p in D.parameters()])
sum([p.flatten().shape[0] for p in D.parameters()])

D(torch.randn(1000,28,28)).max()
X = torch.load('output/wgan/data/data_param_mnist_pct_0_iter_0.pt')
X.min()




sns.displot(D(X).detach().numpy(), stat='probability')


img = X

img_flat = lambda img: img.view(img.shape[0], -1)

layer = lambda img_flat:torch.nn.Linear(int(np.prod(D.IMG_SHAPE)), 512)(img_flat).detach().cpu().numpy()
for i in range(100):
    val = layer(img_flat(img))
    val.shape, val.min(), val.max()
    if val.max()>3:print(i, val.max())

layer1 = torch.nn.Linear(int(np.prod(D.IMG_SHAPE)), 512)
act = torch.nn.LeakyReLU(0.2, inplace=True)
layer2 = torch.nn.Linear(512, 256)
# torch.nn.LeakyReLU(0.2, inplace=True),


sns.displot(val.flatten(), stat='probability')

val2 = layer2(act(layer1(img_flat(X)))).detach().numpy().flatten()
sns.displot(val2, stat='probability')

sns.displot(val.flatten(), stat='probability')

l = [m for m in D.modules()]

l[1][1]
import torch.nn as nn
model = nn.Sequential(
    # https://stackoverflow.com/questions/32514502/neural-networks-what-does-the-input-layer-consist-of
    nn.Linear(int(np.prod(D.IMG_SHAPE)), 512), # hidden layer 1
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(512, 256), # hidden layer 2
    nn.LeakyReLU(0.2, inplace=True),
    nn.Linear(256, 1), # output layer
)

model = nn.Linear(512, 256)
[v.flatten().mean() for v in model.state_dict().values()]
