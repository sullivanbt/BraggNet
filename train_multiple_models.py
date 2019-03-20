import time
time.sleep(7200)

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()
import pickle
import pySlice
from tqdm import tqdm
from skimage.filters import threshold_otsu
from skimage import measure
import pandas as pd

from keras.models import Model, load_model, save_model
import mltools
reload(mltools)



#Initialization
np.random.seed(42)

#baseDirectory = '/data/ml_peak_sets/peaks_tf_mltoolstest_limitedNoise_0p025_cutoff_0p5MaxNoise/'
#baseDirectory = '/data/dna_0p025_cutoff_0p5MaxNoise/'
#baseDirectory = '/data/ml_peak_sets/beta_lac_firstxtal/'
#baseDirectory = '/data/ml_peak_sets/beta_lac_secondcrystal_0p4qMask/'
#baseDirectory = '/data/ml_peak_sets/beta_lac_secondcrystal_0p4qMask_0p15peakThreshold/'
baseDirectory = '/data/dna_0p025_cutoff_0p5MaxNoise_pfCutoff/'
useQMask = True
#=============================================================================================
#Read the training data in
X_train, Y_train, X_test, Y_test = mltools.readDataForTraining(baseDirectory, useQMask=useQMask)
"""
baseDirectory = '/data/ml_peak_sets/beta_lac_secondcrystal_0p4qMask_0p15peakThreshold/'
X_train_1, Y_train_1, X_test_1, Y_test_1 = mltools.readDataForTraining(baseDirectory, useQMask=useQMask, maxNumPeaksTrain=5000)
baseDirectory = '/data/ml_peak_sets/beta_lac_firstxtal/'
X_train_2, Y_train_2, X_test_2, Y_test_2 = mltools.readDataForTraining(baseDirectory, useQMask=useQMask, maxNumPeaksTrain=5000)
X_train = np.append(X_train_1, X_train_2,axis=0)
Y_train = np.append(Y_train_1, Y_train_2,axis=0)
X_test = np.append(X_test_1, X_test_2,axis=0)
Y_test = np.append(Y_test_1, Y_test_2,axis=0)
"""
for i in range(1,12+1):
    print 'Training model {}'.format(i)
    #=============================================================================================
    #Setup the Unet in Keras using TF backend
    reload(mltools)
    model = mltools.build_unet(doBatchNormalization=True)
    model.summary()
    shuffleIDX = np.random.permutation(len(X_train))
    model.fit(X_train[shuffleIDX], Y_train[shuffleIDX], epochs=100, validation_split=0.1)

    #=============================================================================================
    # Save the history
    hist = model.history.history
    plt.figure(12)
    plt.clf()
    for key in np.sort(hist.keys()):
        plt.plot(hist[key], label=key)
    plt.legend(loc='best')

    model.save('/home/ntv/ml_peak_integration/models/dna_{}.h5'.format(i))
    pickle.dump(shuffleIDX, open('/home/ntv/ml_peak_integration/models/dna_{}_shuffleIDX.pkl'.format(i),'wb'))
    pickle.dump(hist, open('/home/ntv/ml_peak_integration/models/dna_{}_hist.pkl'.format(i),'wb'))
#=============================================================================================
# Do some quick evaluation
thresh = 0.4
bulkMasksTrain = mltools.getPeakMaskBulk(X_train, model,thresh=thresh)   
bulkMasksTest = mltools.getPeakMaskBulk(X_test, model,thresh=thresh)  
dcIm = mltools.diceCoeffPerImage(Y_train.squeeze(), bulkMasksTrain.squeeze())
dcImTest = mltools.diceCoeffPerImage(Y_test.squeeze(), bulkMasksTest.squeeze())
dcList = np.append(dcIm, dcImTest)
iouIm = mltools.iouPerImage(Y_train.squeeze(), bulkMasksTrain.squeeze(), thresh=thresh)
iouImTest = mltools.iouPerImage(Y_test.squeeze(), bulkMasksTest.squeeze(), thresh=thresh)
try:
    dfSim = pd.DataFrame(pickle.load(open(baseDirectory+'simulated_peak_params.pkl')))
    dfSim['dcIm'] = dcList
    dfSim['IoU'] = np.append(iouIm, iouImTest)
except:
    print 'unet_keras::Training data were generated in parallel.  Need to work out how to create dfSim'

#=============================================================================================
'''
# Show some results
ix = np.random.randint(0, len(X_test) - 1)
n_events = np.array(X_test[ix]).squeeze()
X_disp = X_test[ix].squeeze()
Y_disp = bulkMasksTest[ix].squeeze()
pySlice.simpleSlices(X_disp, Y_disp/Y_disp.max()*n_events.max())
'''

ix = np.random.randint(0, len(X_train) - 1)
n_events = np.array(X_train[ix]).squeeze()
X_disp = X_train[ix].squeeze()
Y_disp = Y_train[ix].squeeze()
pySlice.simpleSlices(X_disp, Y_disp/Y_disp.max()*n_events.max())

