import sys
sys.version_info

#######COMCPUTE EPSILON############
from decimal import Decimal
from gan_archs.wgan import Discriminator
import math
D = Discriminator((1,28,28))
m = min([p.numel() for p in D.parameters() if p.requires_grad and p.numel() > 1])
d = Decimal(1)/Decimal(m)
d
cp = round(d*10**abs(d.adjusted()))/10**-d.adjusted()
cp
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

###########CHECK IMAGES AT EACH EPOCH###########
p = ['mnist', 0, 20, 1.0, True, 'inf', 'earlyStop']
from config_test import *
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

##############VISUALIZE DP R.V.###########
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch

# https://math.stackexchange.com/a/2894689
meta_hook = lambda batch_size, sigma, shape: (1 / batch_size) * sigma * torch.randn(shape).cpu().detach()
plt.plot(meta_hook(64, 0.5, 1000))
plt.show()

sigma = 0.4
eps = 0.04*2*m/M*sqrt(5*log(1/1e-6))/sigma
eps
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-cp*10, cp*10, 0.001)
# Mean = 0, SD = 2.
plt.plot(x_axis, norm.pdf(x_axis,0,(sigma/m)))
x_axis = np.arange(-5, 5, 0.001)
plt.plot(x_axis, norm.pdf(x_axis,0,1))
plt.show()
weights = torch.load('output/dpwgan/data/data_param_mnist_pct_0_iter_0.pt')
# weights = torch.load('tmp/data_param_mnist_pct_0_iter_0.pt')
plt.hist(weights.flatten().cpu().numpy(), 5)
