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
from timeit import default_timer as timer
import sys
import pandas as pd
from tqdm import tqdm
import glob 

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

def getNearestNeighborIndex(x, spl, p=2):
    return np.argmin(getDistanceSquared(x,spl,p=p))

def getDistanceSquared(x, spl, p=2):
    return ((x-spl)**2).sum(axis=1)

def getPeakMask(test_image, peakLibrary, solutions, nX=32, nY=32, nZ=32, p=2):
    gIDX = getNearestNeighborIndex(test_image, peakLibrary, p=2)
    peakIDX = solutions[gIDX].reshape([nX, nY, nZ])
    return peakIDX


#Load our mantid data

#beta lac july 2018 second xtal
peaksFile = '/home/ntv/mandi_preprocessing/beta_lac_july2018/beta_lac_july2018_secondxtal.integrate'
UBFile = '/home/ntv/mandi_preprocessing/beta_lac_july2018/beta_lac_july2018_secondxtal.mat'
DetCalFile = '/home/ntv/mandi_preprocessing/MANDI_June2018.DetCal'
workDir = '/SNS/users/ntv/dropbox/' #End with '/'
nxsTemplate = '/SNS/MANDI/IPTS-8776/nexus/MANDI_%i.nxs.h5'
dQPixel=0.003
q_frame = 'lab'
pplmin_frac=0.9; pplmax_frac=1.1; mindtBinWidth=15; maxdtBinWidth=50
moderatorFile = '/home/ntv/integrate/bl11_moderatorCoefficients_2018.dat'


# Some parameters
peakToGet = 3 #Arbitrary - just has to be less than the number of peaks
padeCoefficients = ICCFT.getModeratorCoefficients(moderatorFile)
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

qMask = ICCFT.getHKLMask(UBMatrix, frac=0.25, dQPixel=dQPixel, dQ=dQ)
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

#Generate the strong peaks library
#---Load 'em
peaksDir = '/data/peaks_tf/train/'
peaksFiles = glob.glob(peaksDir+'*pkl')
peaksLibrary = np.zeros([len(peaksFiles), 32*32*32])
solutionLibrary = np.zeros_like(peaksLibrary)
for i, peakFile in tqdm(enumerate(peaksFiles),total=len(peaksFiles)):
    solutionFile = peakFile.replace('train', 'train_solution')
    peaksLibrary[i] = pickle.load(open(peakFile)).ravel()
    solutionLibrary[i] = pickle.load(open(solutionFile)).ravel()
solutionLibrary = solutionLibrary.astype(np.bool)

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
                n_events = box.getNumEventsArray();

                cX, cY, cZ = np.array(n_events.shape)//2
                dX, dY, dZ = nX//2,nY//2,nZ//2
                image = 1.0*n_events - np.median(n_events)
                image = n_events/np.max(image)
                image = image[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]
                image = np.expand_dims(image, axis=3) #should be nX*nY*nZ*1
                n_events_cropped = n_events[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]

                peakMask = getPeakMask(n_events_cropped.ravel(), peaksLibrary, solutionLibrary)

                countsInPeak = np.sum(n_events_cropped[peakMask])
                bgIDX = ~peakMask
                meanBG = n_events_cropped[bgIDX].mean()
                intensity = countsInPeak - meanBG*np.sum(peakMask)
                sigma = np.sqrt(countsInPeak+meanBG*np.sum(peakMask))
                t2 = timer()
                df.at[peakToGet,'IntensML'] = intensity
                df.at[peakToGet,'SigIntML'] = sigma
                df.at[peakToGet,'meanBG'] = meanBG
                df.at[peakToGet,'numVoxelsInPeak'] = np.sum(peakMask)
                print('Old: %4.2f +- %4.2f (%4.2f)'%(peak.getIntensity(), peak.getSigmaIntensity(), peak.getIntensity()/peak.getSigmaIntensity()))
                print('New: %4.2f +- %4.2f (%4.2f)'%(intensity, sigma, intensity/sigma))
                print('Calculated in %4.2f s'%(t2-t1))
            except KeyboardInterrupt:
                0/0
            except:
                #raise
                print('Error with peak %i!'%peakToGet)
               
    df.to_pickle('/home/ntv/Desktop/knn_testing_%i.pkl'%runNumber)

    




