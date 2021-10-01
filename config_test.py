DATASET=('mnist',)
TGT_EPOCHS=(0,)
PCT=(20,)
EPS=(1.0,)
TARGETED=(True,)
ATK = ('inf',)
NOTE=('earlyStop',)

# select GANs to be tested
GAN_CHOICE = ['wgan', 'wgan_gp', 'cgan', 'acgan']

# gan name : epochs, batch size
GAN_SETTINGS = {
'acgan':(1,1000),
'cgan':(1,1000),
'wgan_gp':(1,1000),
'wgan':(1,1000)
}


# GAN_CHOICE = ['acgan']
# GAN_SETTINGS = {
# 'acgan':(50,64),
# 'cgan':(1,1000),
# 'wgan_gp':(1,1000),
# 'wgan':(1,1000)
# }
