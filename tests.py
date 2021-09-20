# %load_ext autoreload
# %autoreload 2

from experiments import Experiment_WGAN, Experiment_WGAN_GP, Experiment_CGAN, Experiment_ACGAN


# TEST NORMAL ARCHS
p = (-1, 0, 20, 1.0, True, 'mnist', 'inf', 'wgangpEarlyStop')
exp =  Experiment_WGAN(epochs=1, verbose=True)
exp.run(p)

exp =  Experiment_WGAN_GP(epochs=1, verbose=True)
exp.run(p)

# TEST CONDITIONAL ARCHS
p = (9, 0, 10, 1.0, True, 'mnist', 'inf', 'wgangpEarlyStop')
exp =  Experiment_CGAN(epochs=1, verbose=True)
exp.run(p)

exp =  Experiment_ACGAN(epochs=1, verbose=True)
exp.run(p)
