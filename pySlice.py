from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, X, Y):
        self.ax = [0,0,0,0]
        self.ax[0] = ax[0][0]
        self.ax[1] = ax[0][1]
        self.ax[2] = ax[1][0]
        self.ax[3] = ax[1][1]
        
        self.ax[0].set_title('Data')
        self.ax[1].set_title('Model')
        self.ax[2].set_title('Data - Model')
        self.ax[3].set_title('Tricolor')

        self.X = X
        self.Y = Y
        self.Z = X-Y
        self.W = np.empty(X.shape+(3,))
        self.W[:,:,:,0] = np.zeros_like(X)
        self.W[:,:,:,1] = X
        self.W[:,:,:,2] = Y
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = self.ax[0].imshow(self.X[:, :, self.ind],interpolation='None')
        self.im1 = self.ax[1].imshow(self.Y[:, :, self.ind],interpolation='None')
        self.im2 = self.ax[2].imshow(self.Z[:, :, self.ind],interpolation='None')
        self.im3 = self.ax[3].imshow(self.W[:, :, self.ind],interpolation='None')
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def onclose(self, event):
        print('Closed figure!')
        self.isClosed = True

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.im.set_clim([0,np.max(self.X)])
        #self.im.set_cmap('jet')
        #self.im1.set_cmap('jet')
        nX,nY,nZ = self.X.shape

        self.im1.set_data(self.Y[:, :, self.ind])
        self.im1.set_clim([0,np.max(self.X)])

        self.im2.set_data(self.Z[:, :, self.ind])
        self.im2.set_clim([-np.max(self.X),np.max(self.X)])
        #self.im2.set_clim([0,1])

        self.im3.set_data(self.W[:, :, self.ind])

        self.ax[0].set_ylabel('slice %s' % self.ind)

#        for i in range(4):
#            self.ax[i].set_xlim([0.3*nX,0.7*nX])
#            self.ax[i].set_ylim([0.3*nY,0.7*nY])
#            self.ax[i].set_xlim([20,60])
#            self.ax[i].set_ylim([80,120])
#            self.ax[i].set_xlim([90,110])
#            self.ax[i].set_ylim([90,110])
#            self.ax[i].set_xlim([155,180])
#            self.ax[i].set_ylim([155,180])
        
        self.im.axes.figure.canvas.draw()
        self.im1.axes.figure.canvas.draw()

class ProductTracker(IndexTracker):
    def __init__(self, ax, X, Y, W):
        self.ax = [0,0,0,0]
        self.ax[0] = ax[0][0]
        self.ax[1] = ax[0][1]
        self.ax[2] = ax[1][0]
        self.ax[3] = ax[1][1]

        self.ax[0].set_title('Set 1')
        self.ax[1].set_title('Set 2')
        self.ax[2].set_title('Set 1 * Set 2')
        self.ax[3].set_title('Data')

        self.X = X/np.max(X)
        self.Y = Y/np.max(Y)
        self.Z = 1.0*X/np.max(X)*Y/np.max(Y)
        self.W = W
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = self.ax[0].imshow(self.X[:, :, self.ind],interpolation='None')
        self.im1 = self.ax[1].imshow(self.Y[:, :, self.ind],interpolation='None')
        self.im2 = self.ax[2].imshow(self.Z[:, :, self.ind],interpolation='None')
        self.im3 = self.ax[3].imshow(self.W[:, :, self.ind],interpolation='None')
        self.update()

class SimpleTracker(IndexTracker):
    def __init__(self, ax, X, Y, shapeRatio):
        self.ax = [0,0]
        self.ax[0] = ax[0]
        self.ax[1] = ax[1]

        self.ax[0].set_title('Set 1')
        self.ax[1].set_title('Set 2')

        self.X = X
        self.Y = Y
       
        self.sliceRatio = int(shapeRatio[2])
        self.isClosed = False
        
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = self.ax[0].imshow(self.X[:, :, self.ind],interpolation='None')
        self.im1 = self.ax[1].imshow(self.Y[:, :, self.ind*self.sliceRatio],interpolation='None')
        self.update()
 
    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.im.set_clim([0,np.max(self.X)])
        #self.im.set_cmap('jet')
        #self.im1.set_cmap('jet')
        nX,nY,nZ = self.X.shape

        self.im1.set_data(self.Y[:, :, self.ind*self.sliceRatio])
        self.im1.set_clim([0,np.max(self.X)])
        
        picShape = [0,0]
        picShape[0] = self.X.shape
        picShape[1] = self.Y.shape 
        #for i in range(2):
        #    self.ax[i].set_xlim([0.4*picShape[i][0], 0.6*picShape[i][0]])
        #    self.ax[i].set_ylim([0.4*picShape[i][1], 0.6*picShape[i][1]])


def showSlices(X,Y,Z=None,showProduct=False):
    fig, ax = plt.subplots(2,2)
    #X = np.random.rand(20, 20, 40)
    if not showProduct:
        tracker = IndexTracker(ax, X,Y)
    else:
        tracker = ProductTracker(ax, X,Y,Z)

    while True:
        try:
            fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
            plt.pause(0.01)
        except:
            return 
    plt.show()

def simpleSlices(X,Y):
    shapeRatio = 1.*np.array(Y.shape) / np.array(X.shape)
    if not np.all(shapeRatio == shapeRatio.astype(int)):
        print('NOT INTEGRAL DIFFERENCE')
        return
    fig, ax = plt.subplots(1,2)
    tracker = SimpleTracker(ax, X,Y, shapeRatio)
    while not tracker.isClosed:
        try:
            fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
            fig.canvas.mpl_connect('close_event', tracker.onclose)
            plt.pause(0.01)
        except:
            return 
    plt.show()
