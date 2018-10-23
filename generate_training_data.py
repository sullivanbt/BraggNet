import sys
#remove the original mantid path

popList = []
for i in range(len(sys.path))[::-1]:
    if 'antid' in sys.path[i]:
        sys.path.pop(i)
import socket
if 'sns' in socket.gethostname():
    sys.path.append('/SNS/users/ntv/mantid/mantid/release/bin')
    #sys.path.append('/SNS/users/ntv/workspace/mantid/release/bin')
else: 
    #sys.path.append('/home/ntv/mantid/mantid/bin/')
    sys.path.append('/home/ntv/workspace/mantid/release/bin/')

import matplotlib.pyplot as plt
plt.ion()
#if '../' not in sys.path: sys.path.append('../')
import numpy as np
from scipy.optimize import curve_fit
from mantid.simpleapi import *
from mantid.kernel import V3D
import ICCFitTools as ICCFT
import BVGFitTools as BVGFT
import pickle
from scipy.ndimage import convolve, rotate
reload(ICCFT)
reload(BVGFT)
import pandas as pd
from tqdm import tqdm
import glob
import os


peakNumbersToGet = [15]

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

SetInstrumentParameter(Workspace='peaks_ws', ParameterName='fitConvolvedPeak', ParameterType='Bool', Value='False')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='sigX0Scale', ParameterType='Number', Value='1.0')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='sigY0Scale', ParameterType='Number', Value='1.0')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numDetRows', ParameterType='Number', Value='255')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numDetCols', ParameterType='Number', Value='255')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numBinsTheta', ParameterType='Number', Value='50')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='numBinsPhi', ParameterType='Number', Value='50')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='fracHKL', ParameterType='Number', Value='0.4')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='dQPixel', ParameterType='Number', Value='0.003')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='mindtBinWidth', ParameterType='Number', Value='5.0')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='maxdtBinWidth', ParameterType='Number', Value='50.0')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='peakMaskSize', ParameterType='Number', Value='5')
SetInstrumentParameter(Workspace='peaks_ws', ParameterName='iccKConv', ParameterType='String', Value='100.0 140.0 120.0')

dQPixel = peaks_ws.getInstrument().getNumberParameter("dQPixel")[0]
instrumentName = peaks_ws.getInstrument().getFullName()
mindtBinWidth = peaks_ws.getInstrument().getNumberParameter("minDTBinWidth")[0]
maxdtBinWidth = peaks_ws.getInstrument().getNumberParameter("maxDTBinWidth")[0]
nTheta = peaks_ws.getInstrument().getIntParameter("numBinsTheta")[0]
nPhi   = peaks_ws.getInstrument().getIntParameter("numBinsPhi")[0]
iccFitDict = ICCFT.parseConstraints(peaks_ws)
strongPeakParams = None

# Clean the output directories
fileList = glob.glob('/data/peaks_tf/train/*pkl')
fileList += glob.glob('/data/peaks_tf/train_solution/*pkl')
fileList += glob.glob('/data/peaks_tf/test/*pkl')
fileList += glob.glob('/data/peaks_tf/test_solution/*pkl')
print 'Removing %i files'%len(fileList)
for fileName in fileList:
    try:
        os.remove(fileName)
    except:
        print('Could not remove %s'%fileName)



df = pd.DataFrame(peaks_ws.toDict())
peakNumbersToGet = df[(df['Intens']>200) & (df['RunNumber']==df['RunNumber'].min())].index.values
numBad = 0
print('Starting to generate output....')
for peakToGet in tqdm(peakNumbersToGet):
    try:
        peak = peaks_ws.getPeak(peakToGet);
        box = ICCFT.getBoxFracHKL(peak, peaks_ws, MDdata, UBMatrix, peakToGet, dQ, fracHKL=0.5, dQPixel=dQPixel,  q_frame=q_frame);
        n_events = box.getNumEventsArray();
        Y3D, gIDX, pp_lambda, params = BVGFT.get3DPeak(peak, peaks_ws, box, padeCoefficients,qMask,nTheta=nTheta, nPhi=nPhi, plotResults=False, zBG=1.96,fracBoxToHistogram=1.0,bgPolyOrder=1, strongPeakParams=strongPeakParams, q_frame=q_frame, mindtBinWidth=mindtBinWidth, pplmin_frac=pplmin_frac, pplmax_frac=pplmax_frac,forceCutoff=200,edgeCutoff=3,maxdtBinWidth=maxdtBinWidth, iccFitDict=iccFitDict);


        n_simulated = n_events.copy()
        Y_simulated = Y3D.copy()
        #Rotate the peak
        theta1, theta2, theta3 = np.random.random(3)*360
        n_simulated = rotate(n_simulated, theta1, axes=(1,0), reshape=False)
        n_simulated = rotate(n_simulated, theta2, axes=(2,0), reshape=False)
        n_simulated = rotate(n_simulated, theta3, axes=(2,1), reshape=False)
        Y_simulated = rotate(Y_simulated, theta1, axes=(1,0), reshape=False)
        Y_simulated = rotate(Y_simulated, theta2, axes=(2,0), reshape=False)
        Y_simulated = rotate(Y_simulated, theta3, axes=(2,1), reshape=False)

        
        #We're going to downsample to 32**3 - let's just go a little off center to train mispredicted peaks
        #Downsample to 32**3
        maxOffset = 3
        nVoxelsPerSide = 32
        nX, nY, nZ = np.array(n_events.shape)
        cX, cY, cZ = np.array(n_events.shape)//2
        cX += np.random.randint(low=-1*maxOffset,high=maxOffset)
        cY += np.random.randint(low=-1*maxOffset,high=maxOffset)
        cZ += np.random.randint(low=-1*maxOffset,high=maxOffset)
        dX, dY, dZ = nVoxelsPerSide//2, nVoxelsPerSide//2, nVoxelsPerSide//2
        n_simulated = n_simulated[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]
        Y_simulated = Y_simulated[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]        
        qMask_simulated = qMask[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]
        #Grab our peak shape
        peakIDX = Y_simulated/Y_simulated.max() > 0.025;

        
        #Add noise to the n_simulated
        bgNoiseLevel = 20
        YNoise = np.random.poisson(lam=np.random.random()*bgNoiseLevel, size=n_simulated.shape)
        n_simulated = n_simulated + YNoise
        n_simulated = n_simulated*qMask_simulated
        
        #Expand to 64 bits
        #n_simulated = np.pad(n_simulated, [[(64-nX)//2,(64-nX)//2+nX%2],[(64-nY)//2,(64-nY)//2+nY%2],[(64-nZ)//2,(64-nZ)//2+nZ%2]],'constant',constant_values=0.)
        #peakIDX = np.pad(peakIDX, [[(64-nX)//2,(64-nX)//2+nX%2],[(64-nY)//2,(64-nY)//2+nY%2],[(64-nZ)//2,(64-nZ)//2+nZ%2]],'constant',constant_values=False)

        #Write the answer
        if np.random.rand() > 0.1:
            pickle.dump(n_simulated, open('/data/peaks_tf/train/%i.pkl'%peakToGet,'wb'));
            pickle.dump(peakIDX,  open('/data/peaks_tf/train_solution/%i.pkl'%peakToGet,'wb'));
        else:
            pickle.dump(n_simulated, open('/data/peaks_tf/test/%i.pkl'%peakToGet,'wb'));
            pickle.dump(peakIDX,  open('/data/peaks_tf/test_solution/%i.pkl'%peakToGet,'wb'));

    except KeyboardInterrupt:
        raise
    except:
        #raise
        numBad += 1
    
