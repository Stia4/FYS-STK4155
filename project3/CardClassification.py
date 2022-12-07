import sys; sys.path.append("src") # Importing relative codes with relative imports ends badly, adding folder manually
from src.NeuralNetwork import NeuralNetwork # Still using relative position notation (src.) to get VSCode highlighting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
seed = 0

####################################################
# Data loading and processing

# types = {'spades':1, 'hearts':2, 'clubs':3, 'diamonds':4}
# numbers = {'ace':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7,
#            'eight':8, 'nine':9, 'ten':10, 'jack':11, 'queen':12, 'king':13}

## Read table with card data
dir = 'data/'
cards = pd.read_csv(dir+'cards.csv')

## DATASETS:
## Binary: 1 for red cards, 0 for black
t_suit = np.array(np.logical_or(cards['suit'] == 2, cards['suit'] == 4), dtype=int)
## Binary: 1 for ace of spades, 0 for eight of hearts, need separate cards set since not all fall into these groups
cards_duo = cards.loc[np.logical_or(np.logical_and(cards['num'] == 1, cards['suit'] == 1),  # ace of spades
                                    np.logical_and(cards['num'] == 8, cards['suit'] == 2))] # eight of hearts
t_duo = np.array(np.logical_and(cards_duo['suit'] == 1, cards_duo['num'] == 1), dtype=int)

## CHOOSE DATASET:
cards = cards_duo
t = t_duo

## Load all images and store as design matrix
def load_images(cards):
    image_size = Image.open(dir+np.array(cards['path'])[0]).size # Grab first image, assume it represents rest
    X = np.zeros((len(cards.values), *image_size))
    for i, path in enumerate(cards['path']):
        img = Image.open(dir+path)
        X[i] = np.array(img) / 255
    return X, image_size

def get_train_test(X, t, cards):
    idx_train = np.where(cards['set'] == 0)
    idx_test  = np.where(cards['set'] != 0) # both train and validation sets

    X_train = X[idx_train]
    X_test  = X[idx_test]
    t_train = t[idx_train]
    t_test  = t[idx_test]
    return X_train, X_test, t_train, t_test

def reduce_size(X, t, n):
    classes = np.unique(t)
    X_new = np.zeros((n*len(classes), *X.shape[1:]))
    t_new = np.zeros(n*len(classes))
    for i, t_i in enumerate(classes):
        idx = np.where(t == t_i)
        X_new[i*n:(i+1)*n] = X[idx][:n]
        t_new[i*n:(i+1)*n] = t[idx][:n]
    return X_new, t_new

X, image_size = load_images(cards)
X_train, X_test, t_train, t_test = get_train_test(X, t, cards)
X_train, t_train = reduce_size(X_train, t_train, 10)
X_test,  t_test  = reduce_size(X_test,  t_test,   3)

####################################################
# Network setup

## Initial Network setup
NN = NeuralNetwork(input_nodes=image_size)
for n_filters in [16, 32, 64]:
    # NN.addLayer(type="Convolutional", params=[3, n_filters, 1, 1], act="ReLU", seed=seed) # [kernel, filters, stride, padding]
    NN.addLayer(type="Convolutional", params=[3, n_filters, 1, 1], act="linear", seed=seed) # [kernel, filters, stride, padding]
    NN.addLayer(type=      "Pooling", params=[2, 'max'], seed=seed)
#NN.addLayer(type="Dense", params=np.prod(image_size),  act="Sigmoid", seed=seed)
NN.addLayer(type="Dense", params= 100,  act="Sigmoid", seed=seed)

NN.addLayer(type="Dense", params=1, act="Sigmoid")
#NN.addLayer(type="Dense", params=1, act="linear")
NN.set_CostFunc("MSE")
#NN.set_CostFunc("MSE")
NN.set_LearningRate(1e-3, method="Adam")
#NN.set_LearningRate(1e-3, method=None)

####################################################
# Training section

print(f"{'Input':20}", image_size, np.prod(image_size))
for l in NN.layers:
    print(f"{l.__class__.__name__:20}", l.output_shape, np.prod(l.output_shape))

#NN.load_params("wb.npy")

X_test = X_train # NB! Using train, allowing overfitting
t_test = t_train

a = NN.test(X_test, classify=0)
cost0 = NN.costfunc(a, t_test)
print(cost0)
print(a)
print(t_test)
a = np.where(a >= 0.5, 1, 0)
print(len(a), sum(a))
print(sum(np.where(a == t_test, 1, 0))/len(t_test))

for i in range(20):
    NN.train(X_train, t_train, epochs=1, seed=seed, silent=False)

    a = NN.test(X_test, classify=0)
    cost1 = NN.costfunc(a, t_test)
    print(a)
    a = np.where(a >= 0.5, 1, 0)
    print(t_test)
    print(cost1)
    print(len(a), sum(a))
    print(sum(np.where(a == t_test, 1, 0))/len(t_test))

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

# shape = (4, 4)
# size = 18/4
# fig, ax = plt.subplots(*shape, sharex=True, sharey=True)
# ax = ax.flatten()
# for i, axi in enumerate(ax):
#     z = np.arange(len(E_Hx))
#     axi.plot(z, E_Hx, label="E_Hx")
#     axi.set_title(f"Snapshot {snap0+i}")
# #sys.stdout = sys.__stdout__

# fig.set_figheight(shape[0] * size)
# fig.set_figwidth(shape[1] * size)

# x0 = int(len(z)/2)
# plt.xlim(x0 - 0.1*x0, x0 + 0.1*x0)
# ax[-1].legend(loc = "lower right")
# ax[0].set_ylabel("Hall field strength")

# fig.tight_layout()
# plt.show()