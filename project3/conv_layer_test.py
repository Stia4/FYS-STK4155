from src.Layers import ConvolutionalLayer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

### Sources for Convolutional layer dev:
### https://cs231n.github.io/convolutional-networks/
### https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
### https://hideyukiinada.github.io/cnn_backprop_strides2.html
### https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
### https://zzutk.github.io/docs/reports/2016.10%20-%20Derivation%20of%20Backpropagation%20in%20Convolutional%20Neural%20Network%20(CNN).pdf

def ff_test():
     ### FEED FORWARD TEST CASE
     k = 3
     n_filters = 2
     stride = 2
     padding = 1
     x = [2,2,0,1,2,0,2,2,0,1,0,0,0,0,2,2,2,0,1,2,1,1,0,2,1,
          0,2,0,0,0,2,1,2,0,1,0,0,2,0,0,0,2,1,0,2,0,2,2,1,0,
          1,1,2,2,2,1,1,0,2,1,2,1,0,1,0,0,1,1,0,0,2,2,0,1,1]
     x = np.array(x).reshape(3,5,5).transpose(1,2,0)
     W0 = np.array([ 1, 0,-1,-1, 1, 0, 0, 1, 1, 0, 1,-1, 0,-1,-1, 0, 1,-1, 0, 1, 0, 0, 1, 1, 1,-1,-1]).reshape(3,3,3).transpose(1,2,0)
     W1 = np.array([-1, 1, 0,-1, 0,-1, 1, 0, 1,-1,-1, 1,-1,-1,-1, 0, 0, 0, 0, 1, 0,-1, 1, 0, 0, 1, 0]).reshape(3,3,3).transpose(1,2,0)
     b0 = 1
     b1 = 0

     act = lambda z: z
     dact = lambda z: np.ones_like(z)

     conv = ConvolutionalLayer(x.shape, k, n_filters, stride, padding, act, dact, 0)
     conv.W[0] = W0
     conv.W[1] = W1
     conv.b[0] = b0
     conv.b[1] = b1

     z, a = conv(x)
     print(z[...,0])
     print(z[...,1])

     # Expected output:
     # [[4. 6. 7.]
     #  [5. 6. 9.]
     #  [0. 1. 4.]]
     # [[  0.  -2.   0.]
     #  [  4.  -2.   1.]
     #  [  3. -14.  -4.]]

     ### IMAGE TESTS
     ### GRAYSCALE IMAGE, 2 FILTERS

     img = Image.open("data/train/ace of clubs/001.png")
     img = np.array(img)

     conv = ConvolutionalLayer(img.shape, k, n_filters, stride, padding, act, dact, 0)
     conv.W[0] = W0[..., 0].reshape(3,3,1)
     conv.W[1] = W1[..., 0].reshape(3,3,1)
     conv.b[0] = b0
     conv.b[1] = b1

     z, a = conv(img)
     plt.imshow(z[...,0])
     plt.show()
     plt.imshow(z[...,1])
     plt.show()

     ### RGB IMAGE, 2 FILTERS

     img = Image.open("raw_data/train/ace of clubs/001.jpg")
     img = np.array(img)

     conv = ConvolutionalLayer(img.shape, k, n_filters, stride, padding, act, dact, 0)
     conv.W[0] = W0
     conv.W[1] = W1
     conv.b[0] = b0
     conv.b[1] = b1

     z, a = conv(img)
     plt.imshow(z[...,0])
     plt.show()
     plt.imshow(z[...,1])
     plt.show()

     ### GRAYSCALE IMAGE, 1 FILTER

     img = Image.open("data/train/ace of clubs/001.png")
     img = np.array(img)

     conv = ConvolutionalLayer(img.shape, k, 1, stride, padding, act, dact, 0)
     conv.W[0] = W0[..., 0].reshape(3,3,1)
     conv.b[0] = b0

     z, a = conv(img)
     plt.imshow(z)
     plt.show()

     ### RGB IMAGE, 1 FILTER

     img = Image.open("raw_data/train/ace of clubs/001.jpg")
     img = np.array(img)

     conv = ConvolutionalLayer(img.shape, k, 1, stride, padding, act, dact, 0)
     conv.W[0] = W0
     conv.b[0] = b0

     z, a = conv(img)
     plt.imshow(a)
     plt.show()

def bp_test():
     ### Backpropogation test case, same as feed forward setup
     ## Setup network
     k = 3
     n_filters = 2
     stride = 2
     padding = 1
     x = [2,2,0,1,2,0,2,2,0,1,0,0,0,0,2,2,2,0,1,2,1,1,0,2,1,
          0,2,0,0,0,2,1,2,0,1,0,0,2,0,0,0,2,1,0,2,0,2,2,1,0,
          1,1,2,2,2,1,1,0,2,1,2,1,0,1,0,0,1,1,0,0,2,2,0,1,1]
     x = np.array(x).reshape(3,5,5).transpose(1,2,0) # Input
     t = np.zeros((3,3,2)) # Targets
     t[..., 0] = [[4, 6, 7], [5, 6, 9], [0, 1, 4]]
     t[..., 1] = [[0, -2, 0], [4, -2, 1], [3, -14, -4]]
     act = lambda z: z # Linear since we compare with z, not a
     dact = lambda z: np.ones_like(z)

     conv = ConvolutionalLayer(x.shape, k, n_filters, stride, padding, act, dact, 0)

     ## Set cost function and learning rate
     Cost = lambda y, t: np.mean((t - y)**2)
     dCost = lambda y, t: (y - t) * 2#/len(t)
     eta0 = 1e-3
     eta = lambda l, lw, g: eta0*g

     ## Train network and print cost
     for i in range(500):
          _, a = conv(x)
          dCda = dCost(a, t)
          conv.update_wb(dCda, eta, 0)
          if not i % 50:
               print(i, Cost(a, t))

ff_test()
bp_test()