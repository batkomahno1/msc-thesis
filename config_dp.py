# FOR DP CAN ONLY CHANGE PCT SIZE OR NUMBER OF DATASETS
# IF USING MORE THAN ONE PERCENTAGE OR EPSILON, CLEAN GAN WILL NOT BE RECALCULATED!!!!!!!!!
DATASET=('mnist','fmnist',)
TGT_EPOCHS=(1,)
PCT=(20,)
EPS=(1.0,)
TARGETED=(True,)
ATK = ('dp','nondp',)
NOTE=('earlyStop',)

# select GAN archs to experiment
GAN_CHOICE = ['dpwgan']

# gan name : epochs, batch size
GAN_SETTINGS = {
'dpwgan':(50,64//4),
}
