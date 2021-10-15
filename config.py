# set experiment params
DATASET=('mnist','fmnist',)
TGT_EPOCHS=(0,)
PCT=(10, 20,)
EPS=(0.0, 1.0,)
TARGETED=(True,)
ATK = ('inf',) # TODO: add other options not implemented error
NOTE=('earlyStop','downgrade',)

# select GAN archs to experiment
GAN_CHOICE = ['wgan', 'wgan_gp', 'cgan', 'acgan']

# gan name : epochs, batch size
GAN_SETTINGS = {
'acgan':(50,1000),
'cgan':(100,500),
'wgan_gp':(50,250),
'wgan':(100,500)
}
