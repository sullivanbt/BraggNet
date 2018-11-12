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

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
set_session(tf.Session(config=config))
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


'''
def getPeakMask(test_image, model):
    test_image = model.predict(test_image).squeeze()
    thresh = threshold_otsu(test_image[5:-5,5:-5,5:-5])
    peakIDX = test_image > thresh
    blobs = measure.label(peakIDX, neighbors=4, background=False)
    peakRegionNumber = np.argmax(np.bincount(blobs[5:-5,5:-5,5:-5].ravel())[1:])+1
    peakIDX = blobs == peakRegionNumber
    return peakIDX
'''

def getPeakMask(test_image_input, model, thresh=0.15, dEdge=8):
    if thresh < 0.: return np.zeros((test_image_input.shape[1:3]),dtype=np.bool)
    try:
        cIM, cX, cY, cZ, cC = np.array(test_image_input.shape)//2
        test_image = model.predict(test_image_input).squeeze()
        peakIDX = test_image > thresh
        blobs = measure.label(peakIDX, neighbors=4, background=False)
        binCounts = np.bincount(blobs[dEdge:-1*dEdge,dEdge:-1*dEdge,dEdge:-1*dEdge].ravel())[1:]
        peakRegionNumber = np.argmax(binCounts)+1
        peakIDX = blobs == peakRegionNumber
    except:
        #raise
        return(getPeakMask(test_image_input, model, thresh=thresh-0.1, dEdge=dEdge))
    return peakIDX

#Define a few things for lsses
def mean_iou(y_true, y_pred):
    smooth = tf.convert_to_tensor(1.)
    # Flatten
    y_true_f = tf.reshape(tf.greater(y_true,0.3), [-1])
    y_pred_f = tf.reshape(tf.greater(y_pred,0.3), [-1])
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(y_true_f, y_pred_f),tf.float32))
    union =        tf.reduce_sum(tf.cast( tf.logical_or(y_true_f, y_pred_f),tf.float32))
    score = (intersection + smooth) / (union + smooth)
    return score

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
model_file = '/home/ntv/ml_peak_integration/models/model_withQMask_fromdfpeaks_relu_halfrot_strongonly_0p5dropout.h5'
#model_file = '/home/ntv/ml_peak_integration/models/model_withQMask_fromdfpeaks_relu_halfrot_allruns_limitNoise_normalizeonly.h5'
#model = load_model(model_file, custom_objects={'dice_coeff':dice_coeff})
model = load_model(model_file, custom_objects={'bce_dice_loss':bce_dice_loss, 'dice_coeff':dice_coeff, 'dice_loss':dice_loss,
                                               'mean_iou':mean_iou})

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


df = pd.DataFrame(peaks_ws.toDict())
df['IntensML'] = np.zeros(len(df))
df['SigIntML'] = np.ones(len(df),dtype=float)
df['meanBG'] = np.zeros(len(df))
df['numVoxelsInPeak'] = np.zeros(len(df))

if len(sys.argv) == 1:
    runNumbers = df['RunNumber'].unique()
else:
    runNumbers = map(int,sys.argv[1:])
print('Integrating run numbers:', runNumbers)
qMask = pickle.load(open('/data/peaks_tf/qMask.pkl', 'rb'))
cX, cY, cZ = np.array(qMask.shape)//2
dX, dY, dZ = nX//2, nY//2, nZ//2
qMaskSimulated = qMask[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]

for runNumber in runNumbers:
    print('Current Run Number %i'%runNumber)

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
        MDdata = ICCFT.getSample(runNumber, DetCalFile, workDir, fileName, q_frame=q_frame)
        MDdata.setComment('BSGETBOX%i'%runNumber)

    #Need for detecting detector edges
    neigh_length_m = 3
    convBox = 1.0*np.ones([neigh_length_m, neigh_length_m,neigh_length_m]) / neigh_length_m**3

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
                if np.sum(peakMask) == 0:
                    print('Peak %i has zero pixels!'%peakToGet)

                #Integration
                n_events_cropped = n_events[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]
                countsInPeak = np.sum(n_events_cropped[peakMask])

                conv_n_events_cropped = convolve(n_events_cropped,convBox)
                bgIDX = reduce(np.logical_and, [~peakMask, qMaskSimulated, conv_n_events_cropped > 0])
                meanBG = n_events_cropped[bgIDX].mean()
                bgCountsInPeak = meanBG*np.sum(peakMask)
                intensity = countsInPeak - bgCountsInPeak
                sigma = np.sqrt(countsInPeak+bgCountsInPeak)
                
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
               
    #df.to_pickle('/home/ntv/Desktop/ml_results/unet_testing_%i_relu_halfrot_strongOnly_0p5dropout.pkl'%runNumber)
    df.to_pickle('/home/ntv/Desktop/ml_results/unet_testing_%i_relu_halfrot_strongOnly_allRuns_limitNoise_normalizeOnly.pkl'%runNumber)

    




