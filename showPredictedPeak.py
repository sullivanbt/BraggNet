from timeit import default_timer as timer
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pickle
sys.path.append('/home/ntv/integrate/analysis/')
import pySlice
from tqdm import tqdm
from skimage.filters import threshold_otsu
from skimage import measure
from scipy.ndimage import convolve, rotate
import sys
import pandas as pd
from tqdm import tqdm
from keras.models import Model, load_model, save_model
from mantid.simpleapi import *
import ICCFitTools as ICCFT


def getPeakMask(test_image, model):
    dEdge = 5
    dC = 8
    cIM, cX, cY, cZ, cC = np.array(test_image.shape)//2
    test_image = model.predict(test_image).squeeze()
    thresh = threshold_otsu(test_image[dEdge:-1*dEdge,dEdge:-1*dEdge,dEdge:-1*dEdge])
    thresh = threshold_otsu(test_image[cX-dC:cX+dC, cY-dC:cY+dC, cZ-dC:cZ+dC])
    peakIDX = test_image > thresh
    blobs = measure.label(peakIDX, neighbors=4, background=False)
    binCounts = np.bincount(blobs[dEdge:-1*dEdge,dEdge:-1*dEdge,dEdge:-1*dEdge].ravel())[1:]
    peakRegionNumber = np.argmax(binCounts)+1
    peakIDX = blobs == peakRegionNumber
    return peakIDX, test_image, blobs, np.array(test_image>thresh,dtype=float)
'''
def getPeakMask(test_image, model):
    test_image = model.predict(test_image).squeeze()
    thresh = threshold_otsu(test_image[5:-5,5:-5,5:-5])
    peakIDX = test_image > thresh
    blobs = measure.label(peakIDX, neighbors=4, background=False)
    peakRegionNumber = np.argmax(np.bincount(blobs[5:-5,5:-5,5:-5].ravel())[1:])+1
    peakIDX = blobs == peakRegionNumber
    return peakIDX, test_image, blobs, np.array(test_image>thresh,dtype=float)
'''
    

#Do some initial stuff for tensorflow
#model_file = 'model_keras.h5' #First pass
#model_file = '/home/ntv/ml_peak_integration/models/model_withQMask_andEmpties.h5'
model_file = '/home/ntv/ml_peak_integration/models/model_withQMask_fromdfpeaks_selu.h5'
if not model:
    print('Loading model from %s'%model_file)
    model = load_model(model_file, custom_objects={'dice_coeff':dice_coeff})

#beta lac july 2018 second xtal
peaksFile = '/home/ntv/mandi_preprocessing/beta_lac_july2018/beta_lac_july2018_secondxtal.integrate'
UBFile = '/home/ntv/mandi_preprocessing/beta_lac_july2018/beta_lac_july2018_secondxtal.mat'
DetCalFile = '/home/ntv/mandi_preprocessing/MANDI_June2018.DetCal'
workDir = '/SNS/users/ntv/dropbox/' #End with '/'
nxsTemplate = '/SNS/MANDI/IPTS-8776/nexus/MANDI_%i.nxs.h5'
dQPixel=0.003
q_frame = 'lab'

nX = 32; nY = 32; nZ = 32

# Some parameters
print('Which peak?')
peakToGet = int(input()) 

importPeaks = True
print('Loading peaks_ws')
for ws in mtd.getObjectNames():
    if mtd[ws].getComment() == '%s'%peaksFile:
        print '    using already loaded peaks file'
        importPeaks = False
        peaks_ws = mtd[ws]
if importPeaks:
    peaks_ws = LoadIsawPeaks(Filename = peaksFile)
    peaks_ws.setComment(peaksFile)
    
LoadIsawUB(InputWorkspace=peaks_ws, FileName=UBFile)
UBMatrix = peaks_ws.sample().getOrientedLattice().getUB()
dQ = np.abs(ICCFT.getDQFracHKL(UBMatrix, frac=0.5))
dQ[dQ>0.2]=0.2

qMask = pickle.load(open('/data/peaks_tf/qMask.pkl', 'rb'))
cX, cY, cZ = np.array(qMask.shape)//2
dX, dY, dZ = nX//2, nY//2, nZ//2
qMaskSimulated = qMask[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]

peak = peaks_ws.getPeak(peakToGet)

#--imports new MDdata
importFlag = True
for ws in mtd.getObjectNames():
    if mtd[ws].getComment() == 'BSGETBOX%i'%peak.getRunNumber():
        print '   Using already loaded MDdata'
        MDdata = mtd[ws]
        importFlag = False
        break
if importFlag:
    fileName = nxsTemplate%peak.getRunNumber()
    MDdata = ICCFT.getSample(peak.getRunNumber(), DetCalFile, workDir, fileName, q_frame=q_frame)
    MDdata.setComment('BSGETBOX%i'%peak.getRunNumber())
    
    
box = ICCFT.getBoxFracHKL(peak, peaks_ws, MDdata, UBMatrix, peakToGet, dQ, fracHKL=0.5, dQPixel=dQPixel,  q_frame=q_frame);
n_events = box.getNumEventsArray();
cX, cY, cZ = np.array(n_events.shape)//2
dX, dY, dZ = nX//2,nY//2,nZ//2
image = n_events[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ] #crop
image = image*qMaskSimulated
image = image / image.max()
image = (image-np.mean(image[qMaskSimulated]))/np.std(image[qMaskSimulated])
image = np.expand_dims(image, axis=3) 
image = np.expand_dims(image, axis=0) #1*nX*nY*nZ*1

'''
#Trial
image = n_events[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ] #crop
image = (image-np.mean(image[image>0]))/np.std(image)
image = np.expand_dims(image, axis=3) #should be nX*nY*nZ*1
'''
peakMask, testim, blobs, threshim = getPeakMask(image, model)

#Integration
n_events_cropped = n_events[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]
countsInPeak = np.sum(n_events_cropped[peakMask])
bgIDX = ~peakMask & qMaskSimulated
meanBG = n_events_cropped[bgIDX].mean()
intensity = countsInPeak - meanBG*np.sum(peakMask)
sigma = np.sqrt(countsInPeak+meanBG*np.sum(peakMask))


z0 = nZ//2
dZ = 2
plt.figure(182)
plt.clf()
plt.subplot(1,2,1)
plt.imshow(image.squeeze()[:,:,z0-dZ:z0+dZ].sum(axis=2))
plt.subplot(1,2,2)
plt.imshow(peakMask[:,:,z0-dZ:z0+dZ].astype(int).sum(axis=2))
plt.suptitle('Peak %i'%peakToGet)

print('Original: %4.2f +- %4.2f (%4.2f)'%(peak.getIntensity(), peak.getSigmaIntensity(),
        peak.getIntensity()/peak.getSigmaIntensity()))
print('New: %4.2f +- %4.2f (%4.2f)'%(intensity, sigma, intensity/sigma))
