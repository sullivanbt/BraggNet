import sys
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pickle
sys.path.append('/home/ntv/integrate/analysis/')
import pySlice
import pandas as pd

from keras.models import load_model
from mantid.simpleapi import *
import ICCFitTools as ICCFT
from scipy.ndimage import convolve, rotate
import mltools
reload(mltools)

#Do some initial stuff for tensorflow
#model_file = 'model_keras.h5' #First pass
#model_file = '/home/ntv/ml_peak_integration/models/model_withQMask_andEmpties.h5'
model_file = '/home/ntv/ml_peak_integration/models/model_withQMask_fromdfpeaks_selu_halfrot.h5'
model_file = '/home/ntv/ml_peak_integration/models/model_withQMask_fromdfpeaks_relu_halfrot_allruns_limitNoise_hklSmallSet.h5'
trainedOnHKL = False
if not model:
    print('Loading model from %s'%model_file)
    model = load_model(model_file, custom_objects={'bce_dice_loss':mltools.bce_dice_loss, 'dice_coeff':mltools.dice_coeff, 
                                               'dice_loss':mltools.dice_loss, 'mean_iou':mltools.mean_iou})

#beta lac july 2018 second xtal
peaksFile = '/home/ntv/mandi_preprocessing/beta_lac_july2018/beta_lac_july2018_secondxtal.integrate'
UBFile = '/home/ntv/mandi_preprocessing/beta_lac_july2018/beta_lac_july2018_secondxtal.mat'
DetCalFile = '/home/ntv/mandi_preprocessing/MANDI_June2018.DetCal'
workDir = '/SNS/users/ntv/dropbox/' #End with '/'
nxsTemplate = '/SNS/MANDI/IPTS-8776/nexus/MANDI_%i.nxs.h5'
dQPixel=0.003
q_frame = 'lab'

# Some parameters
try:
    print('Which peak? (Current is %i)'%peakToGet)
except:
    print('Which peak?')
peakToGet = int(input()) 

peaks_ws = mltools.getPeaksWorkspace(peaksFile, UBFile)
UBMatrix = peaks_ws.sample().getOrientedLattice().getUB()
dQ = np.abs(ICCFT.getDQFracHKL(UBMatrix, frac=0.5))
dQ[dQ>0.2]=0.2

nX = 32; nY = 32; nZ = 32
qMask = pickle.load(open('/data/peaks_tf/qMask.pkl', 'rb'))
cX, cY, cZ = np.array(qMask.shape)//2
dX, dY, dZ = nX//2, nY//2, nZ//2
qMaskSimulated = qMask[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]

peak = peaks_ws.getPeak(peakToGet)

MDdata = mltools.getMDData(peak, nxsTemplate, DetCalFile, workDir, q_frame)    
box = ICCFT.getBoxFracHKL(peak, peaks_ws, MDdata, UBMatrix, peakToGet, dQ, fracHKL=0.5, dQPixel=dQPixel,  q_frame=q_frame);
n_events_cropped, image = mltools.getImageFromBox(box, UBMatrix, peak, rebinToHKL=trainedOnHKL, qMaskSimulated=qMaskSimulated)
peakMask, testim, blobs = mltools.getPeakMask(image, model,  thresh=0.15)
mltools.makeShowPredictedPeakFigure(image, peakMask, peakToGet)

#Integration
countsInPeak = np.sum(n_events_cropped[peakMask])
neigh_length_m = 3
convBox = 1.0*np.ones([neigh_length_m, neigh_length_m,neigh_length_m]) / neigh_length_m**3
conv_n_events_cropped = convolve(n_events_cropped,convBox)
if not trainedOnHKL:
    bgIDX = reduce(np.logical_and, [~peakMask, qMaskSimulated, conv_n_events_cropped > 0])
else:
    bgIDX = np.logical_and(~peakMask, conv_n_events_cropped > 0)
meanBG = n_events_cropped[bgIDX].mean()
bgCountsInPeak = meanBG*np.sum(peakMask)
intensity = countsInPeak - meanBG*np.sum(peakMask)
sigma = np.sqrt(countsInPeak+meanBG*np.sum(peakMask))

print('Original: %4.2f +- %4.2f (%4.2f)'%(peak.getIntensity(), peak.getSigmaIntensity(),
        peak.getIntensity()/peak.getSigmaIntensity()))
print('New: %4.2f +- %4.2f (%4.2f)'%(intensity, sigma, intensity/sigma))
print('Number of voxels in peak: %i'%np.sum(peakMask))
