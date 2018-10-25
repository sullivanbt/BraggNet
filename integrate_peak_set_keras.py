import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()
import pickle
sys.path.append('/home/ntv/integrate/analysis/')
import pySlice
from tqdm import tqdm
from skimage.filters import threshold_otsu
from skimage import measure
from scipy.ndimage import convolve, rotate
from timeit import default_timer as timer
import sys
import pandas as pd
from tqdm import tqdm
from keras.models import Model, load_model, save_model

popList = []
for i in range(len(sys.path))[::-1]:
    if 'antid' in sys.path[i]:
        sys.path.pop(i)
import socket
if 'sns' in socket.gethostname():
    sys.path.append('/SNS/users/ntv/mantid/mantid/release/bin')
else: 
    sys.path.append('/home/ntv/workspace/mantid/release/bin/')

from mantid.simpleapi import *
import ICCFitTools as ICCFT

np.random.seed = 42
nX = 32
nY = 32
nZ = 32
nChannels = 1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def getPeakMask(test_image, model):
    test_image = model.predict(test_image).squeeze()
    thresh = threshold_otsu(test_image[5:-5,5:-5,5:-5])
    peakIDX = test_image > thresh
    blobs = measure.label(peakIDX, neighbors=4, background=False)
    peakRegionNumber = np.argmax(np.bincount(blobs[5:-5,5:-5,5:-5].ravel())[1:])+1
    peakIDX = blobs == peakRegionNumber
    return peakIDX


#Define a few things for lsses
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    from tensorflow.python.keras import losses
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


#Do some initial stuff for tensorflow
#model_file = 'model_keras.h5' #First pass
model_file = '/home/ntv/ml_peak_integration/models/model_withQMask_fromdfpeaks_selu.h5'
model = load_model(model_file, custom_objects={'dice_coeff':dice_coeff})

#Load our mantid data

#beta lac july 2018 second xtal
peaksFile = '/home/ntv/mandi_preprocessing/beta_lac_july2018/beta_lac_july2018_secondxtal.integrate'
UBFile = '/home/ntv/mandi_preprocessing/beta_lac_july2018/beta_lac_july2018_secondxtal.mat'
DetCalFile = '/home/ntv/mandi_preprocessing/MANDI_June2018.DetCal'
workDir = '/SNS/users/ntv/dropbox/' #End with '/'
nxsTemplate = '/SNS/MANDI/IPTS-8776/nexus/MANDI_%i.nxs.h5'
dQPixel=0.003
q_frame = 'lab'


# Some parameters
peakToGet = 3 #Arbitrary - just has to be less than the number of peaks
removeEdges = False 
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


peak = peaks_ws.getPeak(peakToGet)

print('Loading MDdata.  This may take a few minutes.')
importFlag = True
for ws in mtd.getObjectNames():
    if mtd[ws].getComment() == 'BSGETBOX%i'%peak.getRunNumber():
        print '   Using already loaded MDdata'
        MDdata = mtd[ws]
        importFlag = False
        break
if importFlag:
    try:
        fileName = nxsTemplate%peak.getRunNumber()
    except:
        fileName = nxsTemplate.format(0, peak.getRunNumber())
    MDdata = ICCFT.getSample(peak.getRunNumber(), DetCalFile, workDir, fileName, q_frame=q_frame)
    MDdata.setComment('BSGETBOX%i'%peak.getRunNumber())


print('Which Peak?')
peakToGet = int(input())
df = pd.DataFrame(peaks_ws.toDict())
df['IntensML'] = np.zeros(len(df))
df['SigIntML'] = np.ones(len(df),dtype=float)
df['meanBG'] = np.zeros(len(df))
df['numVoxelsInPeak'] = np.zeros(len(df))
runNumbers = df['RunNumber'].unique()

qMask = pickle.load(open('/data/peaks_tf/qMask.pkl', 'rb'))
cX, cY, cZ = np.array(qMask.shape)//2
dX, dY, dZ = nX//2, nY//2, nZ//2
qMaskSimulated = qMask[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]

for runNumber in runNumbers:
    print('Run Number %i'%runNumber)

    #--imports new MDdata
    importFlag = True
    for ws in mtd.getObjectNames():
        if mtd[ws].getComment() == 'BSGETBOX%i'%runNumber:
            print '   Using already loaded MDdata'
            MDdata = mtd[ws]
            importFlag = False
            break
    if importFlag:
        fileName = nxsTemplate%runNumber
        MDdata = ICCFT.getSample(peak.getRunNumber(), DetCalFile, workDir, fileName, q_frame=q_frame)
        MDdata.setComment('BSGETBOX%i'%peak.getRunNumber())

    #---Determine which peaks to try and then run them.
    peakNumbersToGet = df[(df['RunNumber']==runNumber)].index.values

    for peakToGet in tqdm(peakNumbersToGet):
        peak = peaks_ws.getPeak(peakToGet);
        if peak.getRunNumber() == runNumber:
            try:
                t1 = timer()
                box = ICCFT.getBoxFracHKL(peak, peaks_ws, MDdata, UBMatrix, peakToGet, dQ, fracHKL=0.5, dQPixel=dQPixel,  q_frame=q_frame);
                #Get image
                n_events = box.getNumEventsArray();
                cX, cY, cZ = np.array(n_events.shape)//2
                dX, dY, dZ = nX//2,nY//2,nZ//2
                image = n_events[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ] #crop
                image = image*qMaskSimulated
                image = image / image.max()
                image = (image-np.mean(image[qMaskSimulated]))/np.std(image[qMaskSimulated])
                image = np.expand_dims(image, axis=3)
                image = np.expand_dims(image, axis=0)
                peakMask = getPeakMask(image, model)

                #Integration
                n_events_cropped = n_events[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]
                countsInPeak = np.sum(n_events_cropped[peakMask])
                bgIDX = ~peakMask & qMaskSimulated
                meanBG = n_events_cropped[bgIDX].mean()
                intensity = countsInPeak - meanBG*np.sum(peakMask)
                sigma = np.sqrt(countsInPeak+meanBG*np.sum(peakMask))
                
                t2 = timer()
                df.at[peakToGet,'IntensML'] = intensity
                df.at[peakToGet,'SigIntML'] = sigma
                df.at[peakToGet,'meanBG'] = meanBG
                df.at[peakToGet,'numVoxelsInPeak'] = np.sum(peakMask)
                #print('Old: %4.2f +- %4.2f (%4.2f)'%(peak.getIntensity(), peak.getSigmaIntensity(), peak.getIntensity()/peak.getSigmaIntensity()))
                #print('New: %4.2f +- %4.2f (%4.2f)'%(intensity, sigma, intensity/sigma))
                #print('Calculated in %4.2f s'%(t2-t1))
            except KeyboardInterrupt:
                0/0
            except:
                #raise
                print('Error with peak %i!'%peakToGet)
               
    df.to_pickle('/home/ntv/Desktop/ml_results/unet_testing_%i_selu.pkl'%runNumber)

    




