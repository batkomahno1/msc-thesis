from experiment import Experiment
import torch.nn as nn
import torch

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
            self.G_decorator = lambda G: lambda z, kwargs: G(z, kwargs['labels'].to(self.DEVICE))
        if 'D_decorator' not in kwargs.keys():
            self.D_decorator = lambda D: lambda X, kwargs: D(X, kwargs['labels'].to(self.DEVICE))
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
