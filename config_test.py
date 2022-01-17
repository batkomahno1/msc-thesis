# DATASET=('mnist','fmnist',)
# TGT_EPOCHS=(0,)
# PCT=(20,)
# EPS=(0.0, 1.0,) # try zero here
# TARGETED=(True,)
# ATK = ('inf',)
# NOTE=('earlyStop',)
#
# # # select GANs to be tested
# # GAN_CHOICE = ['wgan', 'wgan_gp', 'cgan', 'acgan']
# # GAN_CHOICE = ['acgan', 'cgan', 'wgan', 'wgan_gp']
# GAN_CHOICE = ['wgan']
# # gan name : epochs, batch size
# GAN_SETTINGS = {
# 'acgan':(1,1000),
# 'cgan':(1,1000),
# 'wgan_gp':(1,1000),
# 'wgan':(1,1000)
# }

# check if FIDs are same as during prototyping
DATASET=('mnist','fmnist',)
TGT_EPOCHS=(0,)
PCT=(20,)
EPS=(0.0, 1.0,) # try zero here
TARGETED=(True,)
ATK = ('inf',)
NOTE=('earlyStop',)
# select GANs to be tested
# GAN_CHOICE = ['wgan_gp', 'cgan']
GAN_CHOICE = ['wgan', 'wgan_gp', 'cgan', 'acgan']
# gan name : epochs, batch size
GAN_SETTINGS = {
'acgan':(50,64),
'cgan':(100,64),
'wgan_gp':(50,64),
'wgan':(100,64),
}
