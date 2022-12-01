import sys; sys.path.append("src") # Importing relative codes with relative imports ends badly, adding folder manually
from src.NeuralNetwork import NeuralNetwork # Still using relative position notation (src.) to get VSCode highlighting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
seed = 0

## Read table with card data
dir = 'data/'
cards = pd.read_csv(dir+'cards.csv')

## Load all images and store as design matrix
image_size = Image.open(dir+cards['path'][0]).size # Grab first image, assume it represents rest
X = np.zeros((len(cards.values), *image_size))
for i, path in enumerate(cards['path']):
    img = Image.open(dir+path)
    X[i] = np.array(img)

# 1 for red cards, 0 for black
t_suit = np.logical_or(np.where(cards['suit'] == 2, 1, 0),
                       np.where(cards['suit'] == 4, 1, 0))

## Initial Network setup
NN = NeuralNetwork(input_nodes=image_size)
NN.addLayer(type="Convolutional", params=[5, 3, 1, 0], act="ReLU", seed=seed)
NN.addLayer(type=      "Pooling", params=[3, 'max'], seed=seed)
NN.addLayer(type="Convolutional", params=[5, 3, 1, 0], act="ReLU", seed=seed)
NN.addLayer(type=      "Pooling", params=[3, 'max'], seed=seed)
NN.addLayer(type=        "Dense", params= 100,  act="Sigmoid", seed=seed)

## Various tests to choose from
def color_test(NN: NeuralNetwork, X: np.ndarray, t: np.ndarray, X_test=None, t_test=None):
    """ Classify black/red cards """
    if not isinstance(X_test, np.ndarray) or not isinstance(t_test, np.ndarray):
        X_test = X
        t_test = t

    NN.addLayer(type="Dense", params=1, act="Sigmoid")
    NN.set_CostFunc("CrossEntropy")
    #NN.set_LearningRate(1e-4, method="Adam")
    NN.set_LearningRate(1e-2, method=None)

    a = NN.classify(X_test)
    cost0 = NN.costfunc(a, t_test)

    NN.train(X, t, epochs=10, seed=seed, silent=False)
    
    a = NN.classify(X_test)
    cost1 = NN.costfunc(a, t_test)
    
    return NN, cost0, cost1

def suit_test(NN: NeuralNetwork, X, t):
    """ Classify based on the 4 suits """
    NN.addLayer(type="Dense", params=4, act="Softmax")

def rank_test(NN: NeuralNetwork, X, t):
    """ Classify based on the 13 ranks """
    NN.addLayer(type="Dense", params=13, act="Softmax")

def full_test(NN: NeuralNetwork, X, t):
    """ Classify all 52 card types """
    #NN.addLayer(type="Dense", n_nodes=13+4, act="???")
    NN.addLayer(type="Dense", params=52, act="Softmax")

## Running
#NN, cost0, cost1 = color_test(NN, X, t_suit)
#suit_test(NN)
#rank_test(NN)
#full_test(NN)

NN.addLayer(type="Dense", params=1, act="Sigmoid")
NN.set_CostFunc("CrossEntropy")
NN.set_LearningRate(1e-3, method="Adam")
#NN.set_LearningRate(1e-2, method=None)

####################################################
# Training section

#NN.load_params("wb.npy")

#a = NN.classify(X)
#cost0 = NN.costfunc(a, t_suit)
#print(cost0)

NN.train(X, t_suit, epochs=10, seed=seed, silent=False)

a = NN.classify(X)
cost1 = NN.costfunc(a, t_suit)
print(cost1)

NN.save_params("wb.npy")

####################################################
# Plotting section

x = NN.layers[0].x
plt.imshow(x)
plt.show()

a = NN.layers[0].a[..., 0]
plt.imshow(a)
plt.show()

a = NN.layers[0].a[..., 1]
plt.imshow(a)
plt.show()

a = NN.layers[0].a[..., 2]
plt.imshow(a)
plt.show()