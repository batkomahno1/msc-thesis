# source: https://github.com/eriklindernoren/PyTorch-GAN.git
import torchvision.transforms as transforms
import torch.nn as nn
import torch.autograd as autograd
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()
        self.IMG_SHAPE = img_shape
        self.LATENT_DIM = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.LATENT_DIM, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.IMG_SHAPE))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.IMG_SHAPE)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.IMG_SHAPE = img_shape

        self.model = nn.Sequential(
            # https://stackoverflow.com/questions/32514502/neural-networks-what-does-the-input-layer-consist-of
            nn.Linear(int(np.prod(self.IMG_SHAPE)), 512), # hidden layer 1
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256), # hidden layer 2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1), # output layer
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
