DATASET=('mnist',)#'fmnist',)
TGT_EPOCHS=(1,)
PCT=(10,)
EPS=(1.0,) # try zero here
TARGETED=(True,)
ATK = ('dp','nondp')
NOTE=('earlyStop',)

# # select GANs to be tested
# GAN_CHOICE = ['wgan', 'wgan_gp', 'cgan', 'acgan']
# GAN_CHOICE = ['acgan', 'cgan', 'wgan', 'wgan_gp']

# # gan name : epochs, batch size
# GAN_SETTINGS = {
# 'acgan':(1,1000),
# 'cgan':(1,1000),
# 'wgan_gp':(1,1000),
# 'wgan':(1,1000)
# }

# check if FIDs are same as during prototyping
GAN_CHOICE = ['dpwgan']
GAN_SETTINGS = {
'acgan':(50,1000),
'cgan':(100,500),
'wgan_gp':(50,250),
'wgan':(100,500//8),
'dpwgan':(50,64//4),
# 'dpwgan':(5,128),
}

1*5*4
64/4
