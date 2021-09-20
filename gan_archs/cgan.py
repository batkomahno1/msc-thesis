# source: https://github.com/eriklindernoren/PyTorch-GAN.git
import torchvision.transforms as transforms
import torch.nn as nn
import torch.autograd as autograd
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim, nb_classes):
        super(Generator, self).__init__()
        self.IMG_SHAPE = img_shape
        self.LATENT_DIM = latent_dim
        self.NB_CLASSES = nb_classes

        self.label_emb = nn.Embedding(self.NB_CLASSES, self.NB_CLASSES)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.LATENT_DIM + self.NB_CLASSES, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.IMG_SHAPE))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.IMG_SHAPE)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape, nb_classes):
        super(Discriminator, self).__init__()
        self.IMG_SHAPE = img_shape
        self.NB_CLASSES = nb_classes

        self.label_embedding = nn.Embedding(self.NB_CLASSES, self.NB_CLASSES)

        self.model = nn.Sequential(
            nn.Linear(self.NB_CLASSES + int(np.prod(self.IMG_SHAPE)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
#             nn.Softmax()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
