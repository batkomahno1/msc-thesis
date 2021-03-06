import os
# # THIS MUST HAPPEN BEFORE TORCH IS IMPORTED!!!
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import argparse
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
# NOTE: I added this
parser.add_argument("--data_path", type=str, default="../../data/mnist", help="data directory")
parser.add_argument("--target_path", type=str, default="../../data/mnist", help="dummy var")
parser.add_argument("--output_id", type=str, default="", help="output identifier")
parser.add_argument("--save_epochs", type=lambda v: v=='True', default=False, help="save weights each epoch")

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

# set gpu
cuda = True if torch.cuda.is_available() else False
if cuda:
    gpu_id = os.environ["CUDA_VISIBLE_DEVICES"][0]
    device = "cuda:" + gpu_id
else:
    device = "cpu"

Tensor = lambda *args: torch.FloatTensor(*args).to(device) if cuda else torch.FloatTensor(*args)

get_dict = lambda v: v.state_dict() if len(os.environ["CUDA_VISIBLE_DEVICES"])<2 else v.module.state_dict()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Move models and losses to device
generator.to(device)
discriminator.to(device)

# Move models to parallel GPUs
if torch.cuda.device_count() > 1:
    print('Running on ', torch.cuda.device_count(), ' GPUs')
    generator, discriminator = [nn.DataParallel(model) for model in [generator, discriminator]]

# NOTE: I added this
# targets don't matter
import torch.utils.data as data_utils
var = torch.load(opt.data_path)
dataset = data_utils.TensorDataset(var, Tensor(torch.ones(var.shape[0])))

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Tensor(imgs)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()

        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)

            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done,
                       nrow=5, normalize=True)
        batches_done += 1

    # NOTE: I added this
    if opt.save_epochs:
        os.makedirs("weights", exist_ok=True)
        torch.save(get_dict(discriminator), './weights/d_'+opt.output_id+'_epoch_'+str(epoch)+'.pth')
        torch.save(get_dict(generator), './weights/g_'+opt.output_id+'_epoch_'+str(epoch)+'.pth')

os.makedirs("weights", exist_ok=True)
name_d = './weights/d_'+opt.output_id+'_epoch_'+str(epoch)+'.pth'
name_g = './weights/g_'+opt.output_id+'_epoch_'+str(epoch)+'.pth'

torch.save(get_dict(discriminator), name_d)
torch.save(get_dict(generator), name_g)
