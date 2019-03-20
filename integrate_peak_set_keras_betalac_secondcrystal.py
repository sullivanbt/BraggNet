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
import mltools

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

#Do some initial stuff for tensorflow
#model_file = 'model_keras.h5' #First pass
#model_file = '/home/ntv/ml_peak_integration/models/model_withQMask_fromdfpeaks_relu_halfrot_strongonly_0p5dropout.h5'
#trainedOnHKL = False; thresh=0.15
#model_file = '/home/ntv/ml_peak_integration/models/model_withQMask_fromdfpeaks_relu_halfrot_allruns_limitNoise_hklFull.h5'
#trainedOnHKL = True; thresh=0.15
#model_file = '/home/ntv/ml_peak_integration/models/beta_lac_secondxtal_0p15peakThreshold_2.h5' #good one
model_file = '/home/ntv/ml_peak_integration/models/beta_lac_mixedTrainingSets_1.h5' #combined
trainedOnHKL = False; thresh=0.15
model = load_model(model_file, custom_objects={'bce_dice_loss':mltools.bce_dice_loss, 'dice_coeff':mltools.dice_coeff, 
                                               'dice_loss':mltools.dice_loss, 'mean_iou':mltools.mean_iou})

#Load our mantid data

#beta lac july 2018 second xtal
peaksFile = '/SNS/users/ntv/dropbox/beta_lac_july2018_secondxtal_mbvg_2/beta_lac_secondcrystal_combined_pf.integrate'
UBFile = '/home/ntv/mandi_preprocessing/beta_lac_july2018/beta_lac_july2018_secondxtal.mat'
DetCalFile = '/home/ntv/mandi_preprocessing/MANDI_June2018.DetCal'
workDir = '/SNS/users/ntv/dropbox/' #End with '/'
nxsTemplate = '/SNS/MANDI/IPTS-8776/nexus/MANDI_%i.nxs.h5'
dQPixel=0.003
q_frame = 'lab'
pplmin_frac=0.9; pplmax_frac=1.1; mindtBinWidth=15; maxdtBinWidth=50
moderatorFile = '/home/ntv/integrate/bl11_moderatorCoefficients_2018.dat'

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
qMask = pickle.load(open('/data/ml_peak_sets/beta_lac_firstxtal/qMask.pkl', 'rb'))
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
        MDdata = ICCFT.getSample(runNumber, DetCalFile, None, fileName, q_frame=q_frame)
        MDdata.setComment('BSGETBOX%i'%runNumber)

    #Need for detecting detector edges
    neigh_length_m = 3
    convBox = 1.0*np.ones([neigh_length_m, neigh_length_m,neigh_length_m]) / neigh_length_m**3

    #---Determine which peaks to try and then run them.
    peakNumbersToGet = df[(df['RunNumber']==runNumber) & (df['DSpacing']>1.2)].index.values

    for peakToGet in tqdm(peakNumbersToGet):
        peak = peaks_ws.getPeak(peakToGet);
        if peak.getRunNumber() == runNumber:
            try:
                #Get image
                box = ICCFT.getBoxFracHKL(peak, peaks_ws, MDdata, UBMatrix, peakToGet, dQ, fracHKL=0.5, dQPixel=dQPixel, q_frame=q_frame);
                n_events_cropped, image = mltools.getImageFromBox(box, UBMatrix, peak, rebinToHKL=trainedOnHKL, qMaskSimulated=qMaskSimulated)
                peakMask, testim, blobs = mltools.getPeakMask(image, model,thresh=thresh)

                #Integration
                countsInPeak = np.sum(n_events_cropped[peakMask])
                conv_n_events_cropped = convolve(n_events_cropped,convBox)
                if not trainedOnHKL:
                    bgIDX = reduce(np.logical_and, [~peakMask, qMaskSimulated])#, conv_n_events_cropped > 0])
                else:
                    bgIDX = np.logical_and(~peakMask, conv_n_events_cropped > 0)
                meanBG = n_events_cropped[bgIDX].mean()
                bgCountsInPeak = meanBG*np.sum(peakMask)
                intensity = countsInPeak - bgCountsInPeak
                sigma = np.sqrt(countsInPeak+bgCountsInPeak)

                #Record the results
                df.at[peakToGet,'IntensML'] = intensity
                df.at[peakToGet,'SigIntML'] = sigma
                df.at[peakToGet,'meanBG'] = meanBG
                df.at[peakToGet,'numVoxelsInPeak'] = np.sum(peakMask)
            except KeyboardInterrupt:
                0/0
            except:
                #raise
                print('Error with peak %i!'%peakToGet)
               
    #df.to_pickle('/home/ntv/Desktop/ml_results/unet_testing_%i_relu_halfrot_strongOnly_0p5dropout_withmltools_noEdgeDetection.pkl'%runNumber)
    #df.to_pickle('/home/ntv/Desktop/ml_results/beta_lac_secondxtal_0p15peakThreshold_2_%i.pkl'%runNumber)
    df.to_pickle('/home/ntv/Desktop/ml_results/beta_lac_secondcrystal_combined_1_%i.pkl'%runNumber)

    




