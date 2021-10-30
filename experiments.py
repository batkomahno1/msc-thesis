import os
from experiment import Experiment
import torch.nn as nn
import torch
import logging
import time
import numpy as np
from experiment import Experiment, _import_model, get_hyper_param, OUTPUT_ID

class Experiment_ACGAN(Experiment):
    def __init__(self, **kwargs):
        super().__init__('acgan', **kwargs)
        # TODO: handle this variable better!!
        self.IMG_SHAPE = (1,32,32)

        # set the subclass defaults
        print()
        if 'G_decorator' not in kwargs.keys():
            self.G_decorator = lambda G: lambda z, kwargs: G(z, kwargs['labels'])
        if 'D_decorator' not in kwargs.keys():
            self.D_decorator = lambda D: lambda X, kwargs: D(X)[0]
        if 'adv_decorator' not in kwargs.keys():
            self.adv_decorator = lambda D: lambda *args: D(args[0])
        # TODO: DOES LOSS FUNCTION NEED TO BE ZEROED OUT OR ANYTHING??!!
        if 'loss' not in kwargs.keys():
            adversarial_loss = torch.nn.BCELoss()
            auxiliary_loss = torch.nn.CrossEntropyLoss()
            loss_ = lambda device: lambda *args: (adversarial_loss.to(device)(args[0][0], args[1]) + \
                                                    auxiliary_loss.to(device)(args[0][1], args[2]))*0.5
            self.loss = loss_(self.DEVICE)

    def _instantiate_G(self):
        if len(self.classes) == 0:
            raise ValueError('Number of classes must be >= 0.')
        return self.G(self.IMG_SHAPE, self.LATENT_DIM, len(self.classes))

    def _instantiate_D(self):
        if len(self.classes) == 0:
            raise ValueError('Number of classes must be >= 0.')
        return self.D(self.IMG_SHAPE, len(self.classes))

class Experiment_CGAN(Experiment):
    def __init__(self, **kwargs):
        super().__init__('cgan', **kwargs)

        # set some default attributes
        if 'G_decorator' not in kwargs.keys():
            self.G_decorator = lambda G: lambda z, kwargs: G(z, kwargs['labels'])#.to(self.DEVICE))
        if 'D_decorator' not in kwargs.keys():
            self.D_decorator = lambda D: lambda X, kwargs: D(X, kwargs['labels'])#.to(self.DEVICE))
        if 'adv_decorator' not in kwargs.keys():
            self.adv_decorator = lambda D: lambda *args: D(args[0], args[1])

    def _instantiate_G(self):
        if len(self.classes) == 0:
            raise ValueError('Number of classes must be >= 0.')
        return self.G(self.IMG_SHAPE, self.LATENT_DIM, len(self.classes))

    def _instantiate_D(self):
        if len(self.classes) == 0:
            raise ValueError('Number of classes must be >= 0.')
        return self.D(self.IMG_SHAPE, len(self.classes))

class Experiment_WGAN(Experiment):
    def __init__(self, **kwargs):
        super().__init__('wgan', **kwargs)

    def _instantiate_G(self):
        return super()._instantiate_G()

    def _instantiate_D(self):
        return super()._instantiate_D()

class Experiment_WGAN_GP(Experiment):
    def __init__(self, **kwargs):
        super().__init__('wgan_gp', **kwargs)

    def _instantiate_G(self):
        return super()._instantiate_G()

    def _instantiate_D(self):
        return super()._instantiate_D()

class Experiment_DPWGAN(Experiment):
    """By default this GAN is non-private!"""
    def __init__(self, **kwargs):
        self.GAN_DIR = os.getcwd() + '/' + 'external/dpwgan/dpwgan/'
        super().__init__('dpwgan', **kwargs)
        self.sigma = None
        self.weight_clip = None
        self.meta_hook = None

    def _instantiate_G(self):
        return super()._instantiate_G()

    def _instantiate_D(self):
        return super()._instantiate_D()

    def _load_raw_data(self, **kwargs):
        """Leave only classes zero and one. Easier to synthesize for DP WGAN!"""
        super()._load_raw_data(kwargs)
        self.classes = [0,1]
        idxs = torch.cat([torch.where(self.targets == tgt)[0].flatten() for tgt in self.classes])
        self.targets = self.targets[idxs]
        self.data = self.data[idxs]

    def _build_gan(self, p, itr=0, save=False):
        logging.info(f'Starting experiment: DPWGAN {p} iteration {itr}.')
        # TODO: use subprocess.check_output instead
        # , sigma=None, weight_clip=0.1, meta_hook=None
        start = time.time()
        c, pct = get_hyper_param(p)

        # check if this is a cln build and copy samples
        if isinstance(p, str):
            X, y = [v.detach().clone().to(self.DEVICE) for v in [self.data, self.targets]]
        else:
            X, y =[v.to(self.DEVICE) for v in self._load_adv_data(p,itr=itr)]

        # prepare noise generator for gan training
        noise_func = lambda n: self.FloatTensor(np.random.normal(0, 1, (n, self.LATENT_DIM)))

        G, D = self._instantiate_G().to(self.DEVICE), self._instantiate_D().to(self.DEVICE)

        if torch.cuda.device_count() > 1:
            devices = range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')))
            G, D = [nn.DataParallel(model, device_ids=devices) for model in [G, D]]

        # get the DP gan trainer
        var = _import_model(self.GAN_NAME.upper(), self.GAN_DIR+self.GAN_NAME+'.py')
        gan_trainer = var(G, D, noise_func)

        # build gan
        # sigma = None => non-private GAN
        gan_trainer.train(X, epochs = self.EPOCHS, n_critics = 5,
                            batch_size=self.BATCH_SIZE, learning_rate=0.00005,
                            sigma=self.sigma, weight_clip=self.weight_clip, meta_hook=self.meta_hook,
                            save_epochs=save, output_id=OUTPUT_ID.format(c,pct,itr), dir=self.GAN_DIR)

        logging.info(f'Processed gan:{c} pct {pct} itr {itr} time {(time.time()-start)//60}m')
