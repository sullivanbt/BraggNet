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
import glob
from scipy.ndimage import convolve, rotate

from keras.models import Model, load_model, save_model
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D
from keras.layers.merge import concatenate

from mantid.simpleapi import *
import ICCFitTools as ICCFT
import BVGFitTools as BVGFT
import ICConvoluted as ICC


def getNumTrueVoxels(Y):
    return Y.sum(axis=tuple(range(1,Y.ndim)))

def generateTrainingPeak(peak, box, Y3D, peakDict, UBMatrix, pRotation=0.5, rebinHKL=False, addNoise=False,
                         maxOffset = 6, nVoxelsPerSide = 32, peakThreshold = 0.025, qMask=None, noiseScaleFactor=1.0):
    #qMask is only used for calculating max events
                           
    n_events = box.getNumEventsArray()
    n_simulated = n_events.copy()
    Y_simulated = Y3D.copy()
    #Rotate the peak
    if(np.random.random()<pRotation):
        theta1, theta2, theta3 = (np.random.random(3)-0.5)*180
        n_simulated = rotate(n_simulated, theta1, axes=(1,0), reshape=False)
        n_simulated = rotate(n_simulated, theta2, axes=(2,0), reshape=False)
        n_simulated = rotate(n_simulated, theta3, axes=(2,1), reshape=False)
        Y_simulated = rotate(Y_simulated, theta1, axes=(1,0), reshape=False)
        Y_simulated = rotate(Y_simulated, theta2, axes=(2,0), reshape=False)
        Y_simulated = rotate(Y_simulated, theta3, axes=(2,1), reshape=False)
        peakDict['theta1'] = theta1
        peakDict['theta2'] = theta2
        peakDict['theta3'] = theta3
    else:
        peakDict['theta1'] = 0.
        peakDict['theta2'] = 0.
        peakDict['theta3'] = 0.
        
    if rebinHKL:
        R = peak.getGoniometerMatrix()
        T = np.linalg.inv(R.dot(UBMatrix))/2/np.pi
        QX, QY, QZ = ICCFT.getQXQYQZ(box)
        H, K, L = T.dot(np.array([QX.ravel(), QY.ravel(), QZ.ravel()])).reshape([3,QX.shape[0], QX.shape[1], QX.shape[2]])
        h0, k0, l0 = peak.getHKL()

        #We're going to downsample to 32**3 - let's just go a little off center to train mispredicted peaks
        #Downsample to 32**3
        hBins = np.linspace(h0-0.8, h0+0.8, nVoxelsPerSide+maxOffset+1)
        kBins = np.linspace(k0-0.8, k0+0.8, nVoxelsPerSide+maxOffset+1)
        lBins = np.linspace(l0-0.8, l0+0.8, nVoxelsPerSide+maxOffset+1)

        useIDX_X = n_events > 0
        nVect = n_events[useIDX_X]
        hVect_X = H[useIDX_X]
        kVect_X = K[useIDX_X]
        lVect_X = L[useIDX_X]

        useIDX_Y = Y3D > 0
        YVect = Y3D[useIDX_Y]
        hVect_Y = H[useIDX_Y]
        kVect_Y = K[useIDX_Y]
        lVect_Y = L[useIDX_Y]

        n_simulated_hkl, edges = np.histogramdd([hVect_X, kVect_X, lVect_X], weights=nVect, bins=np.array([hBins, kBins, lBins]))
        Y_simulated_hkl, edges = np.histogramdd([hVect_Y, kVect_Y, lVect_Y], weights=YVect, bins=np.array([hBins, kBins, lBins]))

        nX, nY, nZ = np.array(n_simulated_hkl.shape)
        cX, cY, cZ = np.array(n_simulated_hkl.shape)//2
        cX += np.random.randint(low=-1*maxOffset,high=maxOffset+1)
        cY += np.random.randint(low=-1*maxOffset,high=maxOffset+1)
        cZ += np.random.randint(low=-1*maxOffset,high=maxOffset+1)
        dX, dY, dZ = nVoxelsPerSide//2, nVoxelsPerSide//2, nVoxelsPerSide//2
        
        lowX = cX-dX
        highX = cX+dX
        lowY = cY-dY
        highY = cY+dY
        lowZ = cZ-dZ
        highZ = cZ+dZ
        if lowX < 0:
            lowX = 0
            highX = 2*dX
        elif highX > n_simulated_hkl.shape[0]:
            highX = n_simulated_hkl.shape[0]
            lowX = highX - 2*dX
        
        if lowY < 0:
            lowY = 0
            highY = 2*dY
        elif highY > n_simulated_hkl.shape[1]:
            highY = n_simulated_hkl.shape[1]
            lowY = highY - 2*dY
        
        if lowZ < 0:
            lowZ = 0
            highZ = 2*dZ
        elif highZ > n_simulated_hkl.shape[2]:
            highZ = n_simulated_hkl.shape[2]
            lowZ = highZ - 2*dZ
        peakDict['cX'] = cX
        peakDict['cY'] = cY
        peakDict['cZ'] = cZ                         
        n_simulated = n_simulated_hkl[lowX:highX, lowY:highY, lowZ:highZ]
        Y_simulated = Y_simulated_hkl[lowX:highX, lowY:highY, lowZ:highZ]  

    else: #Do in reciprocal space (not HKL)
        nX, nY, nZ = np.array(n_events.shape)
        cX, cY, cZ = np.array(n_events.shape)//2
        cX += np.random.randint(low=-1*maxOffset,high=maxOffset+1)
        cY += np.random.randint(low=-1*maxOffset,high=maxOffset+1)
        cZ += np.random.randint(low=-1*maxOffset,high=maxOffset+1)
        dX, dY, dZ = nVoxelsPerSide//2, nVoxelsPerSide//2, nVoxelsPerSide//2
        
        lowX = cX-dX
        highX = cX+dX
        lowY = cY-dY
        highY = cY+dY
        lowZ = cZ-dZ
        highZ = cZ+dZ
        if lowX < 0:
            lowX = 0
            highX = 2*dX
        elif highX > n_simulated.shape[0]:
            highX = n_simulated.shape[0]
            lowX = highX - 2*dX
        
        if lowY < 0:
            lowY = 0
            highY = 2*dY
        elif highY > n_simulated.shape[1]:
            highY = n_simulated.shape[1]
            lowY = highY - 2*dY
        
        if lowZ < 0:
            lowZ = 0
            highZ = 2*dZ
        elif highZ > n_simulated.shape[2]:
            highZ = n_simulated.shape[2]
            lowZ = highZ - 2*dZ

        peakDict['cX'] = cX
        peakDict['cY'] = cY
        peakDict['cZ'] = cZ                         
        n_simulated = n_simulated[lowX:highX, lowY:highY, lowZ:highZ]
        Y_simulated = Y_simulated[lowX:highX, lowY:highY, lowZ:highZ]
        
        peakIDX = Y_simulated/Y_simulated.max() > peakThreshold
        if qMask is not None:
            cX, cY, cZ = np.array(qMask.shape)//2
            dX, dY, dZ = nVoxelsPerSide//2, nVoxelsPerSide//2, nVoxelsPerSide//2
            qMask_simulated = qMask[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]       
            peakDict['maxEvents'] = n_simulated[peakIDX].max()
            peakDict['bgEvents'] = n_simulated[qMask_simulated & ~peakIDX].mean()
        else:
            peakDict['maxEvents_noQMask'] = n_simulated[peakIDX].max()
            peakDict['bgEvents_noQMask'] = n_simulated[qMask_simulated & ~peakIDX].mean()        
        
    #Add noise to the n_simulated
    if addNoise:
        #bgNoiseLevel = 10.#ICCFT.get_pp_lambda(n_simulated, n_simulated>0)[0]
        if qMask is not None:
            bgNoiseLevel = n_simulated[peakIDX].max()*noiseScaleFactor
        else:
            bgNoiseLevel = n_simulated.max()*noiseScaleFactor
            print('Warning! mltools::generateTrainingPeak has no qMask.  Might add a lot of noise!')
        pp_lambda = np.random.random()*bgNoiseLevel
        YNoise = np.random.poisson(lam=pp_lambda, size=n_simulated.shape)
        peakDict['noiseAdded'] = pp_lambda
        n_simulated = n_simulated + YNoise
    else:
        peakDict['noiseAdded'] = 0. 

    return n_simulated, Y_simulated, peakDict


def cleanOutputDirectoriesForNewTrainingData(baseDirectory):
    if not os.path.exists(baseDirectory):
        print('Creating path %s'%(baseDirectory))
        os.mkdir(baseDirectory)
    for pathType in ['train/', 'train_solution/', 'test/', 'test_solution/']:
        if not os.path.exists(baseDirectory+pathType):
            print('Creating path %s'%(baseDirectory+pathType))
            os.mkdir(baseDirectory+pathType)
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

def getMDData(peak, nxsTemplate, DetCalFile, workDir, q_frame):
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
    return MDdata

def getPeaksWorkspace(peaksFile, UBFile):
    importPeaks = True
    for ws in mtd.getObjectNames():
        if mtd[ws].getComment() == '%s'%peaksFile:
            print '   Using already loaded peaks file'
            importPeaks = False
            peaks_ws = mtd[ws]
            break
    if importPeaks:
        peaks_ws = LoadIsawPeaks(Filename = peaksFile)
        peaks_ws.setComment(peaksFile)
        LoadIsawUB(InputWorkspace=peaks_ws, FileName=UBFile)
    return peaks_ws

def makeShowPredictedPeakFigure(image, peakMask, peakToGet, figNumber=182):
    nX, nY, nZ = image.squeeze().shape
    z0 = nZ//2
    dZ = 2
    plt.figure(figNumber)
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(image.squeeze()[:,:,z0-dZ:z0+dZ].sum(axis=2))
    plt.subplot(1,2,2)
    plt.imshow(peakMask[:,:,z0-dZ:z0+dZ].astype(int).sum(axis=2))
    plt.suptitle('Peak %i'%peakToGet)

def getImageFromBox(box, UBMatrix,peak, qMaskSimulated=None, rebinToHKL=False, nVoxelsPerSide=32, hklRebinFrac=0.8):
    n_events = box.getNumEventsArray();
    if rebinToHKL:
        R = peak.getGoniometerMatrix()
        T = np.linalg.inv(R.dot(UBMatrix))/2/np.pi
        QX, QY, QZ = ICCFT.getQXQYQZ(box)
        H, K, L = T.dot(np.array([QX.ravel(), QY.ravel(), QZ.ravel()])).reshape([3,QX.shape[0], QX.shape[1], QX.shape[2]])
        h0, k0, l0 = peak.getHKL()
        
        hBins = np.linspace(h0-hklRebinFrac, h0+hklRebinFrac, nVoxelsPerSide+1)
        kBins = np.linspace(k0-hklRebinFrac, k0+hklRebinFrac, nVoxelsPerSide+1)
        lBins = np.linspace(l0-hklRebinFrac, l0+hklRebinFrac, nVoxelsPerSide+1)

        useIDX_X = n_events > 0
        nVect = n_events[useIDX_X]
        hVect_X = H[useIDX_X]
        kVect_X = K[useIDX_X]
        lVect_X = L[useIDX_X]

        image, edges = np.histogramdd([hVect_X, kVect_X, lVect_X], weights=nVect, bins=np.array([hBins, kBins, lBins]))
        n_events_cropped = image.copy()
        image = image / image.max()
        image = (image-np.mean(image))/np.std(image)
        image = np.expand_dims(image, axis=3) 
        image = np.expand_dims(image, axis=0) #1*nX*nY*nZ*1
    else:
        cX, cY, cZ = np.array(n_events.shape)//2
        dX, dY, dZ = nVoxelsPerSide//2,nVoxelsPerSide//2,nVoxelsPerSide//2
        image = n_events[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ] #crop
        if qMaskSimulated is not None:
            image = image*qMaskSimulated
        n_events_cropped = n_events[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]
        image = image / image.max()
        if qMaskSimulated is not None:
            image = (image-np.mean(image[qMaskSimulated]))/np.std(image[qMaskSimulated])
        else:
            image = (image-np.mean(image))/np.std(image)
        image = np.expand_dims(image, axis=3) 
        image = np.expand_dims(image, axis=0) #1*nX*nY*nZ*1    

    return n_events_cropped, image



def reconstructY3D(box, df, peak, peakToGet):
    goodEntry = df['PeakNumber'] == peakToGet
    XBOX = BVGFT.boxToTOFThetaPhi(box, peak)
    #TOF
    XTOF = XBOX[:, :, :, 0]
    fICC = ICC.IkedaCarpenterConvoluted()
    fICC.init()
    fICC['A'] = float(df[goodEntry]['Alpha'])
    fICC['B'] = float(df[goodEntry]['Beta'])
    fICC['R'] = float(df[goodEntry]['R'])
    fICC['T0'] = float(df[goodEntry]['T0'])
    fICC['HatWidth'] = 0.5
    fICC['KConv'] = float(df[goodEntry]['KConv'])
    fICC['Scale'] = float(df[goodEntry]['Scale'])

    x_lims = [(1 - 0.015) * peak.getTOF(),
              (1 + 0.015) * peak.getTOF()]
    YTOF = BVGFT.getYTOF(fICC, XTOF, x_lims)

    #BVG
    XTHETA = XBOX[:,:,:,1]
    XPHI = XBOX[:,:,:,2]

    mu = [float(df[goodEntry]['MuTH']), float(df[goodEntry]['MuPH'])]
    sigX = float(df[goodEntry]['SigX'])
    sigY = float(df[goodEntry]['SigY'])
    sigP = float(df[goodEntry]['SigP'])
    sigma = np.array([[sigX**2, sigP * sigX * sigY], [sigP * sigX * sigY, sigY**2]])
    YBVG = BVGFT.bvg(1.0, mu, sigma, XTHETA, XPHI, 0)
    
    Y3D = YBVG*YTOF/YBVG.max()/YTOF.max()*float(df[goodEntry]['scale3d'])
    return Y3D

def readDataForTraining(baseDirectory, nX=32, nY=32, nZ=32, nChannels=1, useQMask=False, maxNumPeaksTrain=None):
    #Figure out which files to read
    TRAIN_PATH = baseDirectory+'train/'
    TEST_PATH  = baseDirectory+'test/'
    #Set up the qMask
    if useQMask:
        qMask = pickle.load(open(baseDirectory+'qMask.pkl', 'rb'))
        cX, cY, cZ = np.array(qMask.shape)//2
        dX, dY, dZ = nX//2, nY//2, nZ//2
        qMaskSimulated = qMask[cX-dX:cX+dX, cY-dY:cY+dY, cZ-dZ:cZ+dZ]

    # Get train and test IDs
    train_ids = next(os.walk(TRAIN_PATH))[2]
    test_ids = next(os.walk(TEST_PATH))[2]
    
    if maxNumPeaksTrain is not None:
        numTrain = len(train_ids)
        numTest =  len(test_ids)
        fracToKeep = 1.0*maxNumPeaksTrain/numTrain
        if fracToKeep < 1.:
            numTrainToKeep = fracToKeep*numTrain
            numTestToKeep = fracToKeep*numTest
            train_ids = np.random.choice(np.array(train_ids), int(numTrainToKeep))
            test_ids = np.random.choice(np.array(test_ids), int(numTestToKeep))
 
    #=============================================================================================
    # Get and resize train images and masks
    images = np.zeros((len(train_ids), nX, nY, nZ,nChannels), dtype=np.float32)
    labels = np.zeros((len(train_ids), nX, nY, nZ,nChannels), dtype=np.float32)
    print('Getting and resizing train images and masks ... ')
    print('   Directory: %s'%TRAIN_PATH)
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        pathTrain = TRAIN_PATH
        pathTrainSolution = TRAIN_PATH[:-1] + '_solution/'
        image = pickle.load(open(pathTrain+id_,'rb'))
        label = pickle.load(open(pathTrainSolution+id_,'rb'))
        if useQMask:
            image *= qMaskSimulated
            label *= qMaskSimulated
        image = image / image.max()
        if useQMask:
            image = (image-np.mean(image[qMaskSimulated]))/np.std(image[qMaskSimulated])
        else:
            image = (image-np.mean(image))/np.std(image)
        images[n,:,:,:,0] = image
        labels[n,:,:,:,0] = label

    X_train = images
    Y_train = labels

    #=============================================================================================
    # Get and resize train images and masks
    images = np.zeros((len(test_ids), nX, nY, nZ,nChannels), dtype=np.float32)
    labels = np.zeros((len(test_ids), nX, nY, nZ,nChannels), dtype=np.float32)
    print('Getting and resizing test images and masks ... ')
    print('   Directory: %s'%TEST_PATH)
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        pathTest = TEST_PATH
        pathTestSolution = TEST_PATH[:-1] + '_solution/'
        image = pickle.load(open(pathTest+id_,'rb'))
        label = pickle.load(open(pathTestSolution+id_,'rb'))
        if useQMask:
            image *= qMaskSimulated
            label *= qMaskSimulated
        image = image / image.max()
        if useQMask:
            image = (image-np.mean(image[qMaskSimulated]))/np.std(image[qMaskSimulated])
        else:
            image = (image-np.mean(image))/np.std(image)
        images[n,:,:,:,0] = image
        labels[n,:,:,:,0] = label

    X_test = images
    Y_test = labels
    return X_train, Y_train, X_test, Y_test

def build_unet(nX=32, nY=32, nZ=32, nChannels=1, activationName = 'relu', dropoutRate=0.5, doBatchNormalization=False):
    inputs = Input((nX, nY, nZ, nChannels))
    c1 = Conv3D(16, (3,3,3), activation=activationName, kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1) if doBatchNormalization else c1
    c1 = Dropout(dropoutRate)(c1)
    #c1 = Conv3D(16, (3,3,3), activation=activationName, kernel_initializer='he_normal', padding='same')(c1)  #
    c1 = Conv3D(16, (3,3,3), activation=activationName, kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1) if doBatchNormalization else c1
    p1 = MaxPooling3D((2,2,2))(c1)

    c2 = Conv3D(32, (3,3,3), activation=activationName, kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2) if doBatchNormalization else c2
    c2 = Dropout(dropoutRate)(c2)
    #c2 = Conv3D(32, (3,3,3), activation=activationName, kernel_initializer='he_normal', padding='same')(c2)  #
    c2 = Conv3D(32, (3,3,3), activation=activationName, kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2) if doBatchNormalization else c2


    u3 = Conv3DTranspose(16, (2,2,2), strides=(1,1,1), padding='same')(c2)
    u3 = concatenate([c2, u3])
    c3 = Conv3D(16, (3,3,3), activation=activationName,kernel_initializer='he_normal', padding='same')(u3)
    c3 = BatchNormalization()(c3) if doBatchNormalization else c3
    c3 = Dropout(dropoutRate)(c3)
    c3 = Conv3D(16, (3,3,3), activation=activationName, kernel_initializer='he_normal', padding='same')(c3)  #
    c3 = Dropout(dropoutRate)(c3)                                                                            ##
    c3 = BatchNormalization()(c3) if doBatchNormalization else c3
    #c3 = Conv3D(16, (3,3,3), activation=activationName, kernel_initializer='he_normal', padding='same')(c3)  #

    u4 = Conv3DTranspose(16, (2,2,2), strides=(2,2,2), padding='same')(c3)
    u4 = concatenate([c1, u4])
    c4 = Conv3D(16, (3,3,3), activation=activationName ,kernel_initializer='he_normal', padding='same')(u4)
    c4 = Dropout(dropoutRate)(c4)                                                                            ##
    c4 = BatchNormalization()(c4) if doBatchNormalization else c4
    c4 = Conv3D(16, (3,3,3), activation=activationName ,kernel_initializer='he_normal', padding='same')(c4)  #
    c4 = BatchNormalization()(c4) if doBatchNormalization else c4
    #c4 = Conv3D(16, (3,3,3), activation=activationName ,kernel_initializer='he_normal', padding='same')(c4)  #

    outputs = Conv3D(1, (1,1,1), activation='sigmoid')(c4)

    model = Model(inputs=[inputs], outputs=[outputs])

    from keras.optimizers import Adam
    #adamaba = Adam(lr=0.001, beta_1=0.9, beta_2=0.999) #Default
    travis = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999) #Used for first submission, IucrJ 2019
    model.compile(optimizer=travis, loss=dice_loss, metrics=[dice_coeff, mean_iou])
    
    return model

def getPeakMaskBulk(test_image, model, thresh = 0.3):
    dEdge = 8
    cIM, cX, cY, cZ, cC = np.array(test_image.shape)//2
    test_image = model.predict(test_image, verbose=1).squeeze()
    peakIDXTotal = np.zeros(test_image.shape,dtype=np.bool)
    for ii, testim in enumerate((test_image)):
        try:
            peakIDX = testim > thresh
            blobs = measure.label(peakIDX, neighbors=4, background=False)
            binCounts = np.bincount(blobs[dEdge:-1*dEdge,dEdge:-1*dEdge,dEdge:-1*dEdge].ravel())[1:]
            peakRegionNumber = np.argmax(binCounts)+1
            peakIDX = blobs == peakRegionNumber
            peakIDXTotal[ii,...] = peakIDX
        except:pass
    return peakIDXTotal

# returns the peak mask
def getPeakMask(test_image_input, model, thresh=0.15, dEdge=8):
    if thresh < 0.: 
        return (np.zeros((test_image_input.shape[1:-1]),dtype=np.bool), model.predict(test_image_input).squeeze(), 
                np.zeros((test_image_input.shape[1:-1]),dtype=np.bool))
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
        return (np.zeros((test_image_input.shape[1:-1]),dtype=np.bool), model.predict(test_image_input).squeeze(), 
                np.zeros((test_image_input.shape[1:-1]),dtype=np.bool))
    return peakIDX, test_image, blobs

# returns the IoU for each image.  Inputs are np arrays
# of size (n_images, nX, nY, nZ) and the threshold (useful
# if y_pred is a prediction map not a boolean yet).
def iouPerImage(y_true, y_pred, thresh=0.4):
    smooth = 1.
    y_true = y_true > thresh
    y_pred = y_pred > thresh
    intersection = np.logical_and(y_true, y_pred).sum(axis=tuple(range(1,y_true.ndim)))
    union = np.logical_or(y_true, y_pred).sum(axis=tuple(range(1,y_true.ndim)))
    union[union==0]=1
    return 1.*(intersection+smooth)/(union+smooth)

# implementation of mean_iou that can be used as a metric by keras
def mean_iou(y_true, y_pred):
    smooth = tf.convert_to_tensor(1.)
    # Flatten
    y_true_f = tf.reshape(tf.greater(y_true,0.3), [-1])
    y_pred_f = tf.reshape(tf.greater(y_pred,0.3), [-1])
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(y_true_f, y_pred_f),tf.float32))
    union =        tf.reduce_sum(tf.cast( tf.logical_or(y_true_f, y_pred_f),tf.float32))
    score = (intersection + smooth) / (union + smooth)
    return score

# returns the dice coefficient for each image.  Inputs are np arrays
# of size (n_images, nX, nY, nZ).
def diceCoeffPerImage(y_true, y_pred):
    numImages = len(y_true)
    smooth = 1.
    dc = np.zeros(numImages)
    for i in tqdm(range(numImages)):
        intersect = np.logical_and(y_true[i],y_pred[i]).sum()
        dc[i] = (2.*intersect+smooth) / (1.0*y_true[i].sum()+y_pred[i].sum()+smooth)
    return dc

# implementation of the dice coefficient (F1 score) that can be used as a metric by keras
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

# loss function based on the dice coefficient that can be used by keras
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

# combined binary cross entropy and dice loss, can be used as a loss by keras.
def bce_dice_loss(y_true, y_pred):
    from tensorflow.python.keras import losses
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
    
def sigmoid(x):
    return 1. / (1. + np.exp(-x))    
