import os
# # THIS MUST HAPPEN BEFORE TORCH IS IMPORTED!!!
# assert os.environ["CUDA_DEVICE_ORDER"]=="PCI_BUS_ID"
import torch
import torch.optim as optim
import numpy as np
from torchvision import transforms
import math

def adjust_lr(optimizer, cur_lr, decay_rate = 0.1, global_step = 1, rec_iter = 200):

    lr = cur_lr * decay_rate ** (global_step / int(math.ceil(rec_iter * 0.8)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

"""

To get R random different initializations of z from L steps of Gradient Descent.
rec_iter : the number of L of Gradient Descent steps
tec_rr : the number of different random initialization of z

"""

def get_z_sets(model, data, labels, lr, loss, device, rec_iter = 200, rec_rr = 10, input_latent = 64, global_step = 1):

    display_steps = 100

    # the output of R random different initializations of z from L steps of GD
    z_hats_recs = torch.Tensor(rec_rr, data.size(0), input_latent)

    # the R random differernt initializations of z before L steps of GD
    z_hats_orig = torch.Tensor(rec_rr, data.size(0), input_latent)

    # backup labels
    labels_orig = labels['labels'].detach().cpu().clone()

    for idx in range(len(z_hats_recs)):

        z_hat = torch.randn(data.size(0), input_latent).to(device)
        z_hat = z_hat.detach().requires_grad_()

        cur_lr = lr

        optimizer = optim.SGD([z_hat], lr = cur_lr, momentum = 0.7)

        z_hats_orig[idx] = z_hat.cpu().detach().clone()

        for iteration in range(rec_iter):

            optimizer.zero_grad()

            fake_image = model(z_hat, labels)

            # TODO: check if I really need this line
            # print('>>>', fake_image.shape)
            # fake_image = fake_image.view(-1, data.size(1), data.size(2), data.size(3))
            # print(fake_image.shape, '<<<')
            reconstruct_loss = loss(fake_image, data)

            reconstruct_loss.backward()

            optimizer.step()

            cur_lr = adjust_lr(optimizer, cur_lr, global_step = global_step, rec_iter= rec_iter)

        z_hats_recs[idx] = z_hat.cpu().detach().clone()

        # make sure SGD doesn't change labels
        assert all(labels_orig==labels['labels'].cpu().detach().clone())

    return z_hats_orig, z_hats_recs

"""

To get z* so as to minimize reconstruction error between generator G and an image x

"""

def get_z_star(model, data, z_hats_recs, loss, device):

    reconstructions = torch.Tensor(len(z_hats_recs))

    for i in range(len(z_hats_recs)):

        z = model(z_hats_recs[i].to(device))

        z = z.view(-1, data.size(1), data.size(2), data.size(3))

        reconstructions[i] = loss(z, data).cpu().item()

    min_idx = torch.argmin(reconstructions)

    return z_hats_recs[min_idx]


def Resize_Image(target_shape, images):

    batch_size, channel, width, height = target_shape

    Resize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((width,height)),
        transforms.ToTensor(),
    ])

    result = torch.zeros((batch_size, channel, width, height), dtype=torch.float)

    for idx in range(len(result)):
        result[idx] = Resize(images.data[idx])

    return result
