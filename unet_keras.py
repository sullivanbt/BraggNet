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

from keras.models import Model, load_model, save_model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D
from keras.layers.merge import concatenate

#Initialization
np.random.seed = 42
nX = 32
nY = 32
nZ = 32
nChannels = 1
TRAIN_PATH = '/data/peaks_tf/train/'
TEST_PATH  = '/data/peaks_tf/test/'

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[2]
test_ids = next(os.walk(TEST_PATH))[2]

if True:
    #=============================================================================================
    # Get and resize train images and masks
    images = np.zeros((len(train_ids), nX, nY, nZ,nChannels), dtype=np.float32)
    labels = np.zeros((len(train_ids), nX, nY, nZ,nChannels), dtype=np.float32)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        pathTrain = TRAIN_PATH
        pathTrainSolution = TRAIN_PATH[:-1] + '_solution/'
        image = pickle.load(open(pathTrain+id_,'rb'))
        label = pickle.load(open(pathTrainSolution+id_,'rb'))
        image = image / image.max()
        image = image-np.median(image[image>0])
        image/= np.std(image)

        #image = (image-np.mean(image[image>0]))/np.std(image)

        images[n,:,:,:,0] = image
        labels[n,:,:,:,0] = label

    X_train = images
    Y_train = labels

    #=============================================================================================
    # Get and resize train images and masks
    images = np.zeros((len(test_ids), nX, nY, nZ,nChannels), dtype=np.float32)
    labels = np.zeros((len(test_ids), nX, nY, nZ,nChannels), dtype=np.float32)
    print('Getting and resizing test images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        pathTest = TEST_PATH
        pathTestSolution = TEST_PATH[:-1] + '_solution/'
        image = pickle.load(open(pathTest+id_,'rb'))
        label = pickle.load(open(pathTestSolution+id_,'rb'))
        image = image / image.max()
        image = image-np.median(image[image>0])
        image/= np.std(image)

        #image = (image-np.mean(image[image>0]))/np.std(image)


        images[n,:,:,:,0] = image
        labels[n,:,:,:,0] = label

    X_test = images
    Y_test = labels

    print('Done!')


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


#=============================================================================================
#Setup the Unet in Keras using TF backend

inputs = Input((nX, nY, nZ, nChannels))
c1 = Conv3D(16, (3,3,3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = Dropout(0.1)(c1)
c1 = Conv3D(16, (3,3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling3D((2,2,2))(c1)

c2 = Conv3D(32, (3,3,3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv3D(32, (3,3,3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)


u3 = Conv3DTranspose(16, (2,2,2), strides=(1,1,1), padding='same')(c2)
u3 = concatenate([u3, c2])
c3 = Conv3D(16, (3,3,3), activation='elu',kernel_initializer='he_normal', padding='same')(u3)

u4 = Conv3DTranspose(16, (2,2,2), strides=(2,2,2), padding='same')(c3)
u4 = concatenate([u4, c1])
c4 = Conv3D(16, (3,3,3), activation='elu',kernel_initializer='he_normal', padding='same')(u4)

outputs = Conv3D(1, (1,1,1), activation='sigmoid')(c4)


model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.summary()

model.fit(X_train, Y_train,epochs=15, validation_split=0.1)
preds_train = model.predict(X_train, verbose=1)
preds_test = model.predict(X_test, verbose=1)

#=============================================================================================
# Show some results
ix = np.random.randint(0, len(X_test) - 1)
n_events = np.array(X_test[ix]).squeeze()

X_disp = X_test[ix].squeeze()
Y_disp = preds_test[ix].squeeze()

thresh = threshold_otsu(Y_disp[5:-5,5:-5,5:-5])
peakIDX = Y_disp > thresh
blobs = measure.label(peakIDX, neighbors=4, background=False)
peakRegionNumber = np.argmax(np.bincount(blobs.ravel())[1:])+1
peakIDX = 1.0*(blobs == peakRegionNumber)

from scipy.ndimage import convolve
neigh_length_m=3
convBox = 1.0*np.ones([neigh_length_m, neigh_length_m, neigh_length_m]) / neigh_length_m**3
conv_X = convolve(X_disp, convBox)
bgIDX = np.logical_and(peakIDX <1, conv_X != np.median(conv_X))
peakIDX[bgIDX] = 0.5

pySlice.simpleSlices(X_disp, peakIDX/peakIDX.max()*n_events.max())



