import os
# assert os.environ["CUDA_DEVICE_ORDER"]=="PCI_BUS_ID"
import torch
import torch.nn as nn

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import gc
import abc
import sys
import subprocess
import time
import importlib.util
import shutil
import logging
import inspect

# TODO: this is a very hack way to import this libarry
sys.path.append(os.path.join(os.getcwd(), 'external/advertorch'))
from advertorch.attacks import LinfPGDAttack

import importlib
defense_gan = importlib.import_module("external.DefenseGAN-Pytorch.util_defense_GAN")

OUTPUT_ID = 'param_{}_pct_{}_iter_{}'
HYPERPARAM = 'tgt_{}_epoch_{}_eps_{}_tgted_{}_set_{}_atk_{}_note_{}'

IMPLEMENTED_ARCHS = ['cgan','acgan','wgan','wgan_gp', 'dpwgan']
CONDITIONAL_ARCHS = ['cgan', 'acgan']

# THIS IS APPROX 20% OF A SINGLE CLASS
DETECTOR_SAMPLE_SIZE = 1000

def get_hyper_param(p):
    if isinstance(p, str) :
        c, pct = p, 0
    else:
        tgt_class, epoch, pct, eps, targeted, dataset, atk, note = p
        c = HYPERPARAM.format(tgt_class, epoch, eps, targeted, dataset, atk, note)
    return c, pct

def _import_model(model_name, model_path):
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return getattr(model_module, model_name)

to_numpy_squeeze = lambda x: x.detach().squeeze().cpu().numpy()

def _check_mem():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        except:
            pass
    print('Paused for 10 seconds...')
    time.sleep(10)

class Experiment(abc.ABC):
    # TODO: ADD LOGGER
    def __init__(self, gan_name, epochs=100, batch_size=64, device=0, samples_ratained=0.1, \
                    latent_dim=100, nb_samples=1000, G_decorator=None, D_decorator=None, \
                    adv_decorator=None, loss=None, verbose=False):
        # check if GAN arch is implemented
        if gan_name not in IMPLEMENTED_ARCHS:
            raise NotImplementedError('GAN arch not implemented')

        # don't need vars stored in GPU memory anymore, release them!
        torch.cuda.empty_cache()

        self.verbose = verbose

        #setup GPUs
        cuda = device != 'cpu' and torch.cuda.is_available()
        self.DEVICE = 'cuda:'+str(device) if cuda else 'cpu'
        self.FloatTensor = lambda *args: torch.FloatTensor(*args).to(self.DEVICE) if cuda else torch.FloatTensor(*args)
        self.LongTensor = lambda *args: torch.LongTensor(*args).to(self.DEVICE) if cuda else torch.LongTensor(*args)

        self.GAN_NAME = gan_name
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.SAMPLES_RATAINED = samples_ratained
        self.LATENT_DIM = latent_dim
        self.NB_SAMPLES = nb_samples
        self.IMG_SHAPE = (1,28,28)

        # TODO: Create these dirs on init
        # TODO: handle dir names better
        self.DIR = os.getcwd() + '/'
        OUTPUT = self.DIR + 'output/'
        self.DATA_DIR = OUTPUT + self.GAN_NAME + '/data/'
        # TODO: this initialization is hacky
        if 'GAN_DIR' not in self.__dict__:
            self.GAN_DIR = self.DIR + 'external/PyTorch-GAN/implementations/'+self.GAN_NAME+'/'
        self.GAN_WEIGHTS_DIR = self.GAN_DIR + 'weights/'
        self.GAN_ARCHS_DIR = self.DIR + 'gan_archs/'
        self.TEST_MODELS_DIR = self.DIR + 'test_models/'
        self.RESULTS = OUTPUT + self.GAN_NAME + '/results/'
        self.TMP_DIR = self.DIR + 'tmp/'
        self.TEMP_DATA_PATH = self.TMP_DIR + 'data_adv.pt'
        self.TEMP_TARGETS_PATH = self.TMP_DIR + 'targets_adv.pt'
        self.CLN_IMGS_DIR, self.ADV_IMGS_DIR = 'tmp/cln_imgs', 'tmp/adv_imgs'

        self.gan_g_path = self.GAN_WEIGHTS_DIR + 'g_' + OUTPUT_ID + '_epoch_{}.pth'
        self.gan_d_path = self.GAN_WEIGHTS_DIR + 'd_' + OUTPUT_ID + '_epoch_{}.pth'

        # TODO: RENAME MODEL FILE
        # TODO: RENAME VARIABLE
        MNIST_CNN_PATH = self.TEST_MODELS_DIR + 'main.py'# TODO: rename with self
        MNIST_CNN_W_PATH = self.TEST_MODELS_DIR + 'mnist_weights.pt'# TODO: rename with self

        self.CREATED_DIRS = [
        self.DATA_DIR, self.GAN_WEIGHTS_DIR, self.TEST_MODELS_DIR, self.RESULTS, \
        self.TMP_DIR, self.CLN_IMGS_DIR, self.ADV_IMGS_DIR
        ]

        # safely create working dirs
        for directory in self.CREATED_DIRS:
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.data_path = self.DATA_DIR + 'data_' + OUTPUT_ID + '.pt'
        self.targets_path = self.DATA_DIR + 'targets_' + OUTPUT_ID + '.pt'
        self.idxs_path = self.DATA_DIR + 'idxs_' + OUTPUT_ID + '.pt'

        # init data
        self.data = None
        self.targets = None
        self.classes = None

        # init models
        self.D = None
        self.G = None
        self.test_model = None

        # these make sure the num of i/o values interfaces with other class
        # define them in the super class per architecture
        # TODO: I think these are called interfaces, not decorators
        # ALWAYS PASS (X/NOISE, LABELS) TO D_decorator AND G_decorator
        self.G_decorator = lambda G: lambda z, kwargs: G(z) if G_decorator is None else G_decorator
        self.D_decorator = lambda D: lambda X, kwargs: D(X) if D_decorator is None else D_decorator
        # TODO: This one seems to be doing the same thing as D_decorator
        self.adv_decorator = lambda D: lambda *args: D(args[0]) if adv_decorator is None else adv_decorator

        # set default loss
        if loss is None:
            default_loss = nn.BCEWithLogitsLoss()
            self.loss = lambda *args: default_loss.to(self.DEVICE)(args[0], args[1])
        else:
            self.loss = loss(self.DEVICE)

    def _load_raw_data(self, mean=0.5, std=0.5, dataset_name='mnist'):
        # TODO: THIS RESIZING IS A PROBLEM!! FIX IT
        if not all([os.path.isfile(v.format(dataset_name, 0, 0)) for v in [self.data_path, self.targets_path]]):
            if dataset_name=='mnist':
                dataset = datasets.MNIST(
                        '/tmp',
                        train=True,
                        download=True,
                        transform=transforms.Compose([transforms.Resize(self.IMG_SHAPE[-1]),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.5], [0.5])
                                                     ])
                )
            elif dataset_name=='fmnist':
                dataset = datasets.FashionMNIST(
                        '/tmp',
                        train=True,
                        download=True,
                        transform=transforms.Compose([transforms.Resize(self.IMG_SHAPE[-1]),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([mean], [std])
                                                     ])
                )
            else:
                raise NotImplementedError

            # TODO: pass dataloader directly to the GAN
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
            data, targets = next(iter(dataloader))

            if not data.min().item()==-1 and data.max().item()==1:
                raise ValueError('Data Not Clipped')

            torch.save(data, self.data_path.format(dataset_name,0,0))
            torch.save(targets, self.targets_path.format(dataset_name,0,0))

        data, targets = (torch.load(v.format(dataset_name,0,0)) for v in [self.data_path, self.targets_path])
        classes = np.unique(to_numpy_squeeze(targets))
        # self.IMG_SHAPE = data.shape[1:]
        if len(self.IMG_SHAPE) != 3: raise ValueError(f'Wrong Data Shape:{self.IMG_SHAPE}')

        self.data, self.targets, self.classes = data, targets, classes

    def _data_to_CPU(self):
        self.data, self.targets = [v.detach().cpu() for v in [self.data, self.targets]]

    def _data_to_GPU(self):
        self.data, self.targets = [v.to(self.DEVICE) for v in [self.data, self.targets]]

    def _init_gan_models(self):
        # TODO: make D and G names arch dependent!
        path = self.GAN_ARCHS_DIR + self.GAN_NAME + '.py'
        D, G =  (_import_model(v, path) for v in ['Discriminator', 'Generator'])
        if all([inspect.isclass(m) for m in [D,G]]):
            self.D, self.G =  D, G
        else:
            raise ValueError('Model must be an uninstantiated class.')

    @abc.abstractmethod
    def _instantiate_G(self):
        """
        Constructor args depend on GAN arch
        """
        return self.G(self.IMG_SHAPE, self.LATENT_DIM)

    @abc.abstractmethod
    def _instantiate_D(self):
        """
        Constructor args depend on GAN arch
        """
        return self.D(self.IMG_SHAPE)

    def _load_generator(self, c, pct, itr=0, epoch=None, *args):
        if epoch is None: epoch = self.EPOCHS-1
        with torch.no_grad():
            G = self._instantiate_G()
            pretrained_params = torch.load(self.gan_g_path.format(c,pct,itr,epoch),map_location=self.DEVICE)
            G.load_state_dict(pretrained_params, strict=False)
            G.to(self.DEVICE)
            G.eval()
            return G

    def _load_discriminator(self, c, pct, itr=0, epoch=None):
        if epoch is None: epoch = self.EPOCHS-1
        D = self._instantiate_D()
        pretrained_params = torch.load(self.gan_d_path.format(c,pct,itr,epoch), map_location=self.DEVICE)
        D.load_state_dict(pretrained_params, strict=False)
        D.to(self.DEVICE)
        D.eval()
        return D

    # TODO: rename kargs to labels as there is no other possible choice
    def _generate(self, nb_samples, c, pct, itr=0, epoch=None, **kwargs):
        """
        kwargs contain labels only
        """
        if epoch is None: epoch = self.EPOCHS-1

        #load gen
        G = self._load_generator(c, pct, itr=itr, epoch=epoch)

        # sample noise
        noise = self.FloatTensor(np.random.normal(0, 1, (nb_samples, self.LATENT_DIM)))

        # make fakes
        output = self.G_decorator(G)(noise, kwargs).detach().cpu()

        #clean up
        G=G.cpu(); del G, noise

        return output

    # TODO: rename kargs to labels as there is no other possible choice
    def _discriminate(self, X, c, pct, itr=0, epoch=None, **kwargs):
        """
        Input: X, c, pct, itr=0, epoch=None, *args
        Note: Discriminator i/o depends on GAN arch
        """
        if epoch is None: epoch = self.EPOCHS-1
        #load discriminator
        D = self._load_discriminator(c,pct,itr=itr,epoch=epoch)

        # run on decorated D
        output = self.D_decorator(D)(X, kwargs)
        output = to_numpy_squeeze(nn.Sigmoid()(output))

        #clean up
        D=D.cpu(); del D

        return output

    def _make_samples(self, p, itr=0):
        """
        Decorator makes sure that predict function handles i/o correctly in advertorch
        """
        # too expensive in terms of storage to save iterations!
        start = time.time()
        tgt_class, epoch, pct, eps, targeted, dataset, atk, note = p
        c = HYPERPARAM.format(tgt_class, epoch, eps, targeted, dataset, atk, note)

        # get tgt class
        if tgt_class == -1:
            nb_samples = int(pct/100*self.data.shape[0])
            idxs_small_part = np.random.choice(np.arange(self.data.shape[0]), nb_samples, replace=False)
            logging.info(f'Indiscriminate attack - {len(idxs_small_part)} samples')
        else:
            idxs = to_numpy_squeeze(torch.where(self.targets==tgt_class)[0])
            nb_samples = int(pct/100*idxs.shape[0])
            idxs_small_part = np.random.choice(idxs, nb_samples, replace=False)
            logging.info(f'Targeting attack at class {tgt_class} - {len(idxs_small_part)} samples')

        # copy samples
        X, labels = [v.detach().clone().to(self.DEVICE) for v in [self.data, self.targets]]

        # set targets
        y = self.FloatTensor(self.data.shape[0], 1).fill_(0.0).to(self.DEVICE)
        if 'noise' in note.lower():
            X[idxs_small_part] = torch.rand_like(self.data[:idxs_small_part.shape[0]]).to(self.DEVICE)
            y[idxs_small_part] = 1.0
        elif 'earlystop'  in note.lower():
            n, var = idxs_small_part.shape[0], labels[idxs_small_part].to(self.DEVICE).detach().clone()
            X[idxs_small_part] = self._generate(n, dataset, 0, itr=0, epoch=epoch, labels=var).to(self.DEVICE).detach().clone()
            y[idxs_small_part] = 1.0
        elif 'downgrade'  in note.lower():
            y[idxs_small_part] = 0
        else:
            raise NotImplementedError

        #load victim discriminator
        D0 = self._load_discriminator(dataset,0,itr=itr,epoch=self.EPOCHS-1)
        D = self.adv_decorator(D0)

        # set up the atk
        adv = LinfPGDAttack(D, loss_fn=self.loss, clip_min=-1.0, clip_max=1.0, eps=eps, \
                            eps_iter=0.01, nb_iter=100, targeted=targeted)

        # attack samples
        if eps > 0:
            X[idxs_small_part] = adv.perturb(X[idxs_small_part], \
                                                y=y[idxs_small_part],labels=labels[idxs_small_part]).detach().clone()

        # save perturbed samples for GAN poisoning
        torch.save(X[idxs_small_part].cpu(), self.data_path.format(c, pct, itr))
        torch.save(labels[idxs_small_part].cpu(), self.targets_path.format(c, pct, itr))
        torch.save(idxs_small_part, self.idxs_path.format(c, pct, itr))

        # clean up
        self._data_to_CPU()
        del adv, D, D0, X, y, labels, idxs_small_part
        gc.collect()
        torch.cuda.empty_cache()

        # log
        logging.info(f'Processed atk:{c} pct {pct} time {(time.time()-start):0.0f}s')

    def _load_adv_data(self, p, itr=0):
        """Return data and targets containing adv samples."""
        c, pct  = get_hyper_param(p)
        # get idxs of adv samples
        idxs = torch.load(self.idxs_path.format(c, pct, itr))
        # copy clean data
        X, y = [v.detach().clone() for v in [self.data, self.targets]]
        # assgining adv samples
        X[idxs] = torch.load(self.data_path.format(c, pct, itr))
        y[idxs] = torch.load(self.targets_path.format(c, pct, itr))
        return X, y

    def _build_gan(self, p, itr=0, save=False):
        # TODO: use subprocess.check_output instead
        start = time.time()
        c, pct = get_hyper_param(p)

        # check if this is a cln build
        if isinstance(p, str):
            # copy samples
            X_path, y_path = [v.format(c, pct, 0) for v in [self.data_path, self.targets_path]]
        else:
            X, y = self._load_adv_data(p,itr=itr)
            torch.save(X, self.TEMP_DATA_PATH)
            torch.save(y, self.TEMP_TARGETS_PATH)
            X_path, y_path = self.TEMP_DATA_PATH, self.TEMP_TARGETS_PATH

        # run the GAN on psned samples
        proc = subprocess.run(["python3", self.GAN_NAME + ".py",
                               "--target_path=" + y_path,
                               "--img_size=" + str(self.IMG_SHAPE[-1]),
                               "--batch_size=" + str(self.BATCH_SIZE),
                               "--n_epochs=" + str(self.EPOCHS),
                               "--output_id=" + OUTPUT_ID.format(c,pct,itr),
                               "--save_epochs=" + str(save),
                               "--data_path=" + X_path],
                              # capture_output=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              cwd = self.GAN_DIR)
        print(proc.stderr.decode('utf-8'))
        proc.check_returncode()
        # if self.verbose: print(proc.stdout.decode('utf-8'))
        logging.info(f'Processed gan:{c} pct {pct} itr {itr} time {(time.time()-start)//60}m')
        return proc.stdout.decode('utf-8')

    def _plot_D(self, p, itr=0, epoch=None):
        # show effect of adv samples on the victim D
        if epoch is None: epoch = self.EPOCHS-1
        c, pct  = get_hyper_param(p)
        dataset = p[-3]

        #load psnd indices
        idxs_small_part = torch.load(self.idxs_path.format(c,pct,itr))
        idxs_big_part = np.array(list(set(range(self.data.shape[0])) - set(idxs_small_part)))

        #load psnd data and targets
        X, y = [v.to(self.DEVICE) for v in self._load_adv_data(p,itr=itr)]
        self._data_to_GPU()

        # make plots
        if idxs_small_part.shape[0]>0:
            preds_small = self._discriminate(X[idxs_small_part], dataset, 0, itr=itr, \
                                                    epoch=epoch, labels=y[idxs_small_part])
            plt.hist(preds_small, alpha=0.5, label='X adv')
        if idxs_big_part.shape[0]>0:
            preds_big = self._discriminate(self.data[idxs_small_part], dataset, 0, itr=itr, \
                                                    epoch=epoch, labels=self.targets[idxs_small_part])
            plt.hist(preds_big, alpha=0.5, label='X cln')
        plt.legend()
        # plt.title(p)
        name = self.RESULTS + 'PSND_D_'+OUTPUT_ID.format(c, pct, itr)+'.svg'
        plt.savefig(name)
        plt.close()

        # remove from GPU!
        self._data_to_CPU()
        X,y = [v.detach().cpu() for v in [X,y]]
        del X,y

    def _visualize_samples(self, p, itr=0):
        # save images of adv samples
        c, pct = get_hyper_param(p)
        X, _ = self._load_adv_data(p,itr=itr)
        idxs = torch.load(self.idxs_path.format(c,pct,itr))
        name = self.RESULTS + 'ADV_SAMPLES_'+OUTPUT_ID.format(c, pct, itr)+'.png'
        save_image(X[idxs][:25].detach().cpu(), name, nrow=5, normalize=True)

    def check_gan(self, p, itr=0, epoch=None):
        if epoch is None: epoch = self.EPOCHS-1
        c, pct = get_hyper_param(p)
        file = self.gan_g_path.format(c,pct,itr,epoch)
        # print(file)
        return os.path.isfile(file)

    def get_gan_path(self, p, itr=0, epoch=None):
        if epoch is None: epoch = self.EPOCHS-1
        c, pct = get_hyper_param(p)
        return self.gan_g_path.format(c,pct,itr,epoch)

    def make_imgs(self, p, itr=0, nb_samples=2048):
        """Generate images and put them in /tmp/cln and /tmp/adv. Former are originals, later are fake."""
        c, pct = get_hyper_param(p)
        # TODO: hacky when clean GANs are considered
        tgt_class = p[0]

        self._data_to_GPU()
        # Sample labels
        if not isinstance(tgt_class, str) and tgt_class >= 0 and tgt_class in self.classes:
            labels = self.LongTensor(np.array([tgt_class]*nb_samples, dtype=np.int))
        else:
            labels = self.LongTensor(
                [self.classes[np.random.randint(len(self.classes))] for i in range(nb_samples)]
            )
        z_adv = self._generate(nb_samples, c, pct, itr=itr, labels = labels)

        # pick first matching index in targets for each label
        # if no labels, then random picks
        matching = lambda e: torch.where(self.targets==e)[0]
        idxs = [matching(e)[0].item() for e in labels if len(matching(e)) > 0]

        assert all(labels == self.targets[idxs])
        assert len(idxs) == labels.shape[0]

        paths = self.CLN_IMGS_DIR, self.ADV_IMGS_DIR
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

        # used for FID score comparison
        for i in range(nb_samples):
            save_image(self.data[idxs][i].detach().cpu(), f'{paths[0]}/cln_{i}.png', normalize=True)
            save_image(z_adv[i].detach().cpu(), f'{paths[1]}/adv_{i}.png', normalize=True)

        # save some samples for evaluation
        name = self.RESULTS + 'PSND_FAKES_'+OUTPUT_ID.format(c, pct, itr)+'.png'
        save_image(z_adv[:25].detach().cpu(), name, nrow=5, normalize=True)
        name = self.RESULTS + 'ORIGINAL_INPUT_'+OUTPUT_ID.format(c, pct, itr)+'.png'
        save_image(self.data[idxs][:25].detach().cpu(), name, nrow=5, normalize=True)

        # don't need vars stored in GPU memory anymore, release them!
        self._data_to_CPU()
        del z_adv, matching, idxs
        torch.cuda.empty_cache()

    def _measure_FID(self, p, itr=0, nb_samples=2048):
        start = time.time()

        self.make_imgs(p, itr=itr, nb_samples=nb_samples)
        torch.cuda.empty_cache()
        id = os.environ["CUDA_VISIBLE_DEVICES"][-1]
        proc = subprocess.run(["python3",'-m','pytorch_fid','--device','cuda:'+id,
                                self.CLN_IMGS_DIR, self.ADV_IMGS_DIR],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                # capture_output=True,
                                cwd = '.')
        proc.check_returncode()
        fid = float(proc.stdout.decode('utf-8').split()[-1])
        logging.info(f'FID={fid} for {p}_itr_{itr} {(time.time()-start):0.0f}s')
        return fid

    def reset(self):
        """Delete ALL old data"""
        for path in self.CREATED_DIRS:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

    def soft_reset(self):
        """Delete previous results"""
        for path in [self.RESULTS]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

    def download(self, params, itr=0, epoch=None):
        from secrets import SERVER_NAME
        if epoch is None: epoch = self.EPOCHS-1
        c,pct=get_hyper_param(params)
        root_dir = ':dp/msc-thesis/' if self.GAN_NAME == 'dpwgan' else ':msc-thesis/'
        server_path = self.gan_g_path.format(c,pct,itr,epoch).replace(self.DIR, SERVER_NAME+root_dir)
        local_path = self.gan_g_path.format(c,pct,itr,epoch)
        if self.verbose:
            print('server_path:'+server_path)
            print('local_path:'+local_path)
        # sleep to prevent flooding!
        time.sleep(1)
        proc = subprocess.run(["scp", server_path, local_path])
        proc.check_returncode()

    def run(self, params, itr=0, download=False):
        """Returns a FID score. If PSND GAN already exists, it will simply re-do the FID"""
        # start experiment
        logging.info(f'Starting experiment: {self.GAN_NAME} {params} iteration {itr}.')
        start = time.time()

        # don't need vars stored in GPU memory anymore, release them!
        torch.cuda.empty_cache()

        # set dataset
        dataset = params[-3]

        #load data
        if self.verbose: print('Loading data...')
        self._load_raw_data(dataset_name=dataset)
        if self.verbose: print('Data loaded')

        # init GAN models
        self._init_gan_models()
        if self.verbose: print('Models initialized')

        # download clean GAN
        if download: self.download(dataset, itr=itr, epoch=self.EPOCHS-1)

        # check clean gan
        if not self.check_gan(dataset, itr=itr, epoch=self.EPOCHS-1):
            raise ValueError('Clean GAN not found.')

        # download psnd GAN
        if download: self.download(params, itr=itr, epoch=self.EPOCHS-1)

        # check if the GAN was already created
        if not self.check_gan(params, itr=itr, epoch=self.EPOCHS-1):
            # make adv nb_samples
            if self.verbose: print('Crafting adv samples...')
            self._make_samples(params, itr=itr)
            if self.verbose: print('Adv samples created')

            # build adv gan
            if self.verbose: print('Building PSND GAN...')
            self._build_gan(params, itr=itr)
            if self.verbose: print('PSND Gan built')
        else:
            if self.verbose: print('PSND GAN loaded')

        # plot and visualize the results
        self._plot_D(params, itr=itr)
        self._visualize_samples(params, itr=itr)
        if self.verbose: print('Plots and images generated')

        # get fid score
        if self.verbose: print('Calculating FID...')
        fid = self._measure_FID(params, itr=itr)
        if self.verbose: print('FID Calculated')

        # log the experiment
        logging.info(f'Experiment complete. Runtime: {(time.time()-start)//60}m.')

        return fid

    def run_cln(self, dataset, itr=0, download=False):
        """Same as run above, except for clean GANs"""
        # check if the paramter is correct
        if not isinstance(dataset, str):
            raise ValueError('Input must be datset and iteration.')

        # start experiment
        logging.info(f'Starting experiment: {self.GAN_NAME} {dataset} iteration {itr}.')
        start = time.time()

        # TODO: CHECK THIS
        # don't need vars stored in GPU memory anymore, release them!
        torch.cuda.empty_cache()

        #load data
        if self.verbose: print('Loading data...')
        self._load_raw_data(dataset_name=dataset)
        if self.verbose: print('Data loaded')

        # init GAN models
        self._init_gan_models()
        if self.verbose: print('Models initialized')

        # download clean GAN
        if download: self.download(dataset, itr=itr, epoch=self.EPOCHS-1)

        # build clean gan
        # always overwrite clean gan in dp experiments
        if not self.check_gan(dataset, itr=itr, epoch=self.EPOCHS-1) or self.GAN_NAME=='dpwgan':
            if self.verbose: print('Building clean GAN...')
            self._build_gan(dataset, itr=itr, save=True)
            if self.verbose: print('Clean GAN built')
        else:
            if self.verbose: print('Clean GAN loaded')

        # get fid score
        if self.verbose: print('Calculating FID...')
        fid = self._measure_FID(dataset, itr=itr)
        if self.verbose: print('FID Calculated')

        # log the experiment
        logging.info(f'Experiment complete. Runtime: {(time.time()-start)//60}m.')

        return fid

    def detect(self, params, itr=0, download=False, itr_other=None):
        """
        Returns auc, fprs, tprs, thetas of detection and produces a ROC curve.
        For DP uses adv data created by a GAN from a different iteration.
        Otherwise, the GAN might have been overwritten!
        """
        if isinstance(params, str):
            raise ValueError('PSND GAN missing.')

        # check if running detection within the same iteration
        itr_other = itr if itr_other is None else itr_other

        # start experiment
        logging.info(f'Starting detector: {self.GAN_NAME} {params} iteration {itr}.')
        start = time.time()

        if self.verbose: print('Running detector...')

        # don't need vars stored in GPU memory anymore, release them!
        torch.cuda.empty_cache()

        # set dataset
        dataset = params[-3]

        # download clean GAN
        if download: self.download(dataset, itr=itr, epoch=self.EPOCHS-1)

        # check clean gan
        if not self.check_gan(dataset, itr=itr, epoch=self.EPOCHS-1):
            raise ValueError('Clean GAN not found.')

        # init GAN models
        self._init_gan_models()
        if self.verbose: print('Models initialized')

        # prepare a generator with an interface for whatever GAN arch I use
        G0 = self._load_generator(dataset, 0, itr=itr).to(self.DEVICE)
        # TODO: TEST THIS!!
        if torch.cuda.device_count() > 1:
            devices = [int(v) for v in os.environ["CUDA_VISIBLE_DEVICES"].replace(' ', '').split(',')]
            if self.verbose: print('Running on GPUs: ', devices)
            G0 = nn.DataParallel(G0, device_ids=devices)
        G = self.G_decorator(G0)

        # get data with adv samples
        X, y = [v.to(self.DEVICE) for v in self._load_adv_data(params, itr=itr_other)]

        # get indecis of adv samples
        tgt_class, c, pct  = params[0], *get_hyper_param(params)
        adv_idxs = torch.load(self.idxs_path.format(c, pct, itr_other))
        # TODO: VERIFY VALIDITY OF CHOOSING A SUBSET
        nb_samples = min(len(adv_idxs), DETECTOR_SAMPLE_SIZE)
        adv_idxs = np.random.choice(adv_idxs, nb_samples, replace=False)

        if not isinstance(adv_idxs,np.ndarray): raise TypeError('Wrong type: '+type(adv_idxs))

        # find indecis of clean samples
        if tgt_class == -1:
            cln_idxs = np.array(list(set(range(X.shape[0])) - set(adv_idxs)))
        else:
            all_idxs = to_numpy_squeeze(torch.where(y==tgt_class)[0])
            cln_idxs = np.array(list(set(all_idxs) - set(adv_idxs)))

        # randomly select as many cln indecis as adv ones
        cln_idxs = np.random.choice(cln_idxs, len(adv_idxs), replace=False)

        # combine adv and cln idxs
        sel_idxs = np.sort(np.concatenate((cln_idxs,adv_idxs), axis=0))

        # set defgan params
        # TODO: make these class constants?
        learning_rate = 10.0

        # use mse loss b/c comparing images not probabilities/logits
        loss = nn.MSELoss().to(self.DEVICE)

        # get a set of optimal z's
        # TODO: DOUBLE CHECK DOING THIS IN ONE GO VS FOR EACH IMAGE SEPERATELY!!!!!
        z_hats = defense_gan.get_z_sets(G, X[sel_idxs], {'labels':y[sel_idxs]}, learning_rate, loss, \
                                        self.DEVICE, input_latent = self.LATENT_DIM)[-1].to(self.DEVICE)

        # free memory
        torch.cuda.empty_cache()

        # find z* that minimizes losses in X
        with torch.no_grad():
            # # possible memory issues, choose device
            # device = self.DEVICE
            # X, y, z_hats, loss = [v.to(device) for v in [X, y, z_hats, loss]]

            # get losses for different z's
            losses = torch.Tensor([loss(G(z, {'labels':y[sel_idxs]}), X[sel_idxs]).cpu() for z in z_hats])

            # find index of z*
            min_idx = torch.argmin(losses).cpu().item()

            # calculate MSE for each image vs z*
            # TODO: speed this up with matrix operations
            A, B = G(z_hats[min_idx], {'labels':y[sel_idxs]}).squeeze(), X[sel_idxs].squeeze()
            img_losses = torch.Tensor([loss(A[i], B[i]).cpu().item() for i in range(len(A))])

            # normalize losses
            losses /= losses.max()
            img_losses /= img_losses.max()

            # get predictions for different thetas
            thetas = np.linspace(0, 1)
            y_preds = [to_numpy_squeeze((img_losses > theta).to(torch.int)) for theta in thetas]

            # get idxs of adv samples - ground truth
            y_true = torch.zeros_like(y)
            y_true[adv_idxs] = 1
            # remove unsued indecis
            y_true = to_numpy_squeeze(y_true)[sel_idxs]

            # some checks
            assert all(y_pred.shape==y_true.shape for y_pred in y_preds)

            # clean up memory
            del X, y, z_hats, loss
            torch.cuda.empty_cache()

        # get confusion matrices for each theta
        # matrix is like: tn, fp, fn, tp
        # https://stackoverflow.com/a/25009504
        conf_mats = [confusion_matrix(y_true, y_pred).ravel() for y_pred in y_preds]
        fprs = [m[1]/(m[1]+m[0]) for m in conf_mats if len(m) > 1]
        tprs = [m[-1]/(m[-1]+m[-2]) for m in conf_mats if len(m) > 1]

        # calculate AUC
        auc = np.trapz(*(sorted(v) for v in [tprs, fprs]))

        # plot ROC curve
        plt.plot(fprs, tprs, label=dataset+', auc='+str(round(auc,2)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        name = self.RESULTS + 'ROC_'+OUTPUT_ID.format(c, pct, itr)+'.svg'
        plt.savefig(name)
        plt.close()

        # log the experiment
        logging.info(f'Detection complete. AUC:{auc} Runtime: {(time.time()-start)//60}m.')

        return auc, fprs, tprs, thetas
