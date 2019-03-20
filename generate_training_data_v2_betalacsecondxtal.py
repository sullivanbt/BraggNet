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
import ICConvoluted as ICC
import mltools

peakNumbersToGet = [15]

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
peakToGet = 3 #Arbitrary - just has to be less than the number of peaks
padeCoefficients = ICCFT.getModeratorCoefficients(moderatorFile)
removeEdges = False 
importPeaks = True
fractionForTesting = 0.1

for ws in mtd.getObjectNames():
    if mtd[ws].getComment() == '%s'%peaksFile:
        print '    using already loaded peaks file'
        importPeaks = False
        peaks_ws = mtd[ws]
if importPeaks:
    peaks_ws = LoadIsawPeaks(Filename = peaksFile)
    peaks_ws.setComment(peaksFile)



#baseDirectory = '/data/peaks_tf_halfRot_strongOnly_allSets_limitedNoise/'
#baseDirectory = '/data/peaks_tf_mltoolstest_limitedNoise_0p025_cutoff_0p5MaxNoise/'
#baseDirectory = '/data/ml_peak_sets/beta_lac_secondcrystal_0p4qMask/'
baseDirectory = '/data/ml_peak_sets/beta_lac_secondcrystal_0p4qMask_0p15peakThreshold/'


if len(sys.argv) == 1:
    runNumbersToAdd = range(9113,9117+1)
    cleanDirectories = True
else:
    runNumbersToAdd = map(int,sys.argv[1:])
    if 9113 in runNumbersToAdd:
        cleanDirectories = True
    else:
        cleanDirectories = False
print('Will generate data from runs %s'%(str(runNumbersToAdd)))


if not os.path.exists(baseDirectory):
    print('Creating path %s'%(baseDirectory))
    os.mkdir(baseDirectory)
for pathType in ['train/', 'train_solution/', 'test/', 'test_solution/']:
    if not os.path.exists(baseDirectory+pathType):
        print('Creating path %s'%(baseDirectory+pathType))
        os.mkdir(baseDirectory+pathType)
if cleanDirectories:
    # Clean the output directories
    fileList = glob.glob(baseDirectory+'train/*pkl')
    fileList += glob.glob(baseDirectory+'train_solution/*pkl')
    fileList += glob.glob(baseDirectory+'test/*pkl')
    fileList += glob.glob(baseDirectory+'test_solution/*pkl')
    print 'Removing %i files'%len(fileList)
    for fileName in fileList:
        try:
            os.remove(fileName)
        except:
            print('Could not remove %s'%fileName)



LoadIsawUB(InputWorkspace=peaks_ws, FileName=UBFile)
UBMatrix = peaks_ws.sample().getOrientedLattice().getUB()
dQ = np.abs(ICCFT.getDQFracHKL(UBMatrix, frac=0.5))
dQ[dQ>0.2]=0.2

qMask = ICCFT.getHKLMask(UBMatrix, frac=0.4, dQPixel=dQPixel, dQ=dQ)

peak = peaks_ws.getPeak(peakToGet)


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
strongPeakParams = None#pickle.load(open('/home/ntv/integrate/strongPeaksParams_betalac_july2018_secondxtal.pkl','rb'))
padeCoefficients = ICCFT.getModeratorCoefficients('/home/ntv/integrate/bl11_moderatorCoefficients_2018.dat')


df = pd.read_pickle('/home/ntv/Desktop/beta_lac_highres_df.pkl')
numBad = 0
print('Starting to generate output....')
pickle.dump(qMask, open(baseDirectory+'qMask.pkl', 'wb'))
paramList = []

peakThreshold = 0.15

for runNumber in runNumbersToAdd:
    print('   Starting run %i'%(runNumber))
    intensArray = np.array(peaks_ws.column('Intens'))
    strongIDX = intensArray > 200        
    runNumArray = np.array(peaks_ws.column('RunNumber'))
    toFitIDX = np.logical_and(strongIDX, runNumArray == runNumber) 
    peakNumbersToGet = np.where(toFitIDX)[0]

    """
    goodIDX = (df['chiSq'] < 50.0) & (df['chiSq'] > 0.50) & (df['chiSq3d']<4) #& (df['notOutlier']) &  (df['Intens3d'] > -1.*np.inf) 
    dEdge = 1
    edgeIDX = (df['Row'] <= dEdge) | (df['Row'] >= 255-dEdge) | (df['Col'] <= dEdge) | (df['Col'] >= 255-dEdge)
    goodIDX = goodIDX & ~edgeIDX
    strongIDX = goodIDX & (df['Intens3d']>200) & (df['RunNumber']==runNumber)
    strongPeakNumbers = df[strongIDX]['PeakNumber'].values
    strongPeakIndices = df[strongIDX]['PeakNumber'].index
    numStrongPeaks = len(strongPeakNumbers)
    peakNumbersToGet = strongPeakNumbers
    """

    #Reload MDdata if need to for this run
    importFlag = True
    for ws in mtd.getObjectNames():
        if mtd[ws].getComment() == 'BSGETBOX%i'%runNumber:
            print '   Using already loaded MDdata'
            MDdata = mtd[ws]
            importFlag = False
            break
    if importFlag:
        try:
            fileName = nxsTemplate%runNumber
        except:
            fileName = nxsTemplate.format(0, runNumber)
        MDdata = ICCFT.getSample(runNumber, DetCalFile, workDir, fileName, q_frame=q_frame)
        MDdata.setComment('BSGETBOX%i'%runNumber)

    for fitNumber, peakToGet in enumerate(tqdm(peakNumbersToGet)):
        try:
            peakDict = {}
            peak = peaks_ws.getPeak(peakToGet);
            #peak = peaks_ws.getPeak(strongPeakIndices[fitNumber]);
            box = ICCFT.getBoxFracHKL(peak, peaks_ws, MDdata, UBMatrix, peakToGet, dQ, dQPixel=dQPixel,  q_frame=q_frame);
            n_events = box.getNumEventsArray()
            Y3D, gIDX, pp_lambda, params = BVGFT.get3DPeak(peak, peaks_ws, box, padeCoefficients,qMask,nTheta=50, nPhi=50, plotResults=False, zBG=1.96,fracBoxToHistogram=1.0,bgPolyOrder=1, strongPeakParams=None, q_frame='lab', mindtBinWidth=10, pplmin_frac=0.8, pplmax_frac=1.1,forceCutoff=200,edgeCutoff=3,maxdtBinWidth=50);
            #Y3D = mltools.reconstructY3D(box, df[df['RunNumber']==runNumber], peak, peakToGet)
            if(params['chiSq'] > 50 or params['chiSq3d']>4):
                raise ValueError('Could not fit peak {}'.format(peakToGet)) #Bad fit, just move on
            # Set the simulated peaks
            reload(mltools)     
            n_simulated, Y_simulated, peakDict = mltools.generateTrainingPeak(peak, box, Y3D, peakDict, 
                                                 UBMatrix, rebinHKL=False, addNoise=True, peakThreshold=peakThreshold,
                                                 qMask=qMask, noiseScaleFactor=0.5)

            peakIDX = Y_simulated/Y_simulated.max() > peakThreshold;
            
            # Record a few extra things so we can analyze the training peaks if we want
            peakDict['PeakNumber'] = peakToGet
            peakDict['Intens'] = peak.getIntensity()
            peakDict['SigIng'] = peak.getSigmaIntensity()
            peakDict['theta'] = peak.getScattering()*0.5
            peakDict['numVoxelsSimulated'] = peakIDX.sum()
            peakDict['numVoxelRaw'] = (Y3D/Y3D.max() > peakThreshold).sum()

            
            paramList.append(peakDict)
            #Write the answer
            if np.random.rand() > fractionForTesting:
                pickle.dump(n_simulated, open(baseDirectory+'train/%i.pkl'%peakToGet,'wb'));
                pickle.dump(peakIDX,  open(baseDirectory+'train_solution/%i.pkl'%peakToGet,'wb'));
            else:
                pickle.dump(n_simulated, open(baseDirectory+'test/%i.pkl'%peakToGet,'wb'));
                pickle.dump(peakIDX,  open(baseDirectory+'test_solution/%i.pkl'%peakToGet,'wb'));

        except KeyboardInterrupt:
            raise
        except:
            #raise
            print('Error generating training set for peak %i'%peakToGet)
            numBad += 1
    pickle.dump(paramList, open(baseDirectory+'simulated_peak_params_%i.pkl'%runNumber, 'wb'))
