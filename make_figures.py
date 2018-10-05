#Makes a comparison of peaks figure
peakToGet = np.random.choice(peakNumbersToGet)

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

#Grab our peak shape
peakIDX = Y_simulated/Y_simulated.max() > 0.025;


#Add noise to the n_simulated
bgNoiseLevel = 20
YNoise = np.random.poisson(lam=np.random.random()*bgNoiseLevel, size=n_simulated.shape)
n_simulated = n_simulated + YNoise

ccX = n_events.shape[0]//2
ccY = n_events.shape[1]//2
plt.figure(1); plt.clf()
plt.subplot(2,2,1); plt.imshow(n_events[ccX-dX:ccX+dX,ccY-dX:ccY+dX,n_events.shape[2]//2]); plt.title('Measured Events')
plt.axis('off')
plt.subplot(2,2,2); plt.imshow((Y3D/Y3D.max() > 0.025)[ccX-dX:ccX+dX,ccY-dX:ccY+dX,n_events.shape[2]//2]); plt.title('Fitted Profile')
plt.axis('off')
plt.subplot(2,2,3); plt.imshow(n_simulated[:,:,n_simulated.shape[2]//2]); plt.title('Simulated Peak')
plt.axis('off')
plt.subplot(2,2,4); plt.imshow(peakIDX[:,:,n_simulated.shape[2]//2]); plt.title('Simulated Ground Truth')
plt.axis('off')





#Make the metrics figures
plt.figure(1); plt.clf();
plt.subplot(1,2,1);
plt.plot(hist.history['acc'],label='Train Accuracy')
plt.plot(hist.history['val_acc'], label='Validation Accuracy')
plt.legend(loc='best')
plt.subplot(1,2,2);
plt.plot(hist.history['loss'],label='Train Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.legend(loc='best')

