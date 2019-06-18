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
baseDirectory = '/data/ml_peak_sets/beta_lac_secondcrystal_0p4qMask_0p15peakThreshold_folded/'

useQMask = True
#=============================================================================================
#Read the training data in
X_train, Y_train, X_test, Y_test = mltools.readDataForTraining(baseDirectory, useQMask=useQMask)

#Clean any NaNs we got from folding
nanIDX = np.where(np.isnan(X_train).sum(axis=(1,2,3,4)) > 0)[0]
X_train[nanIDX] = 0.
Y_train[nanIDX] = False

#=============================================================================================
#Setup the Unet in Keras using TF backend
reload(mltools)
model = mltools.build_unet(doBatchNormalization=True)
model.summary()
shuffleIDX = np.random.permutation(len(X_train))
model.fit(X_train[shuffleIDX], Y_train[shuffleIDX], epochs=100, validation_split=0.1)

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
# Save the history
hist = model.history.history
plt.figure(12)
plt.clf()
for key in np.sort(hist.keys()):
    plt.plot(hist[key], label=key)
plt.legend(loc='best')
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

