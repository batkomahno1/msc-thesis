# FOR DP CAN ONLY CHANGE PCT SIZE OR NUMBER OF DATASETS
DATASET=('mnist','fmnist',)
TGT_EPOCHS=(1,)
PCT=(20,)
EPS=(1.0,)
TARGETED=(True,)
ATK = ('low','norm')
NOTE=('earlyStop',)

# select GAN archs to experiment
GAN_CHOICE = ['dpwgan']

# gan name : epochs, batch size
GAN_SETTINGS = {
'dpwgan':(100,500),
}

5*4*50//60