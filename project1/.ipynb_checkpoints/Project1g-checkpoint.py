import numpy as np
from imageio.v2 import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from Project1d import kfold
from Project1f import Bootstrap_pred
from tqdm import trange

n_degrees = 10                    # number of degrees, degrees 0 ... n_degrees-1
nk = 4                            # number of k-folds
n_bootstraps = 10                 # number of times to resample each learn-set
lambdas = np.logspace(-5, -1, 5)  # values of lambda to use in Ridge regression

terrain1 = imread('data/SRTM_data_Norway_1.tif')
terrain2 = imread('data/SRTM_data_Norway_2.tif')
z = terrain2[:1000, :1000]
z = terrain2[:250, :250]

plt.imshow(z)
plt.show()

x = np.linspace(0, 1, z.shape[0])
y = np.linspace(0, 1, z.shape[1])
x_, y_ = np.meshgrid(x, y)
learn_set, test_set = kfold(x_, y_, z, nk, seed=0)

for i in range(len(lambdas)):
    lam = lambdas[i]
    bias     = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)
    error    = np.zeros(n_degrees)
    for k in trange(nk):
        x_lk, y_lk, z_lk = learn_set[:, k]
        x_tk, y_tk, z_tk =  test_set[:, k]
        z_pred = Bootstrap_pred(x_lk, x_tk, y_lk, y_tk, z_lk, z_tk, n_bootstraps, n_degrees, lam)
        for degree in range(n_degrees):
            error[degree]    += np.mean( np.mean((z_tk[:, np.newaxis] - z_pred[degree])**2, axis=1, keepdims=True) ) / nk
            bias[degree]     += np.mean( (z_tk[:, np.newaxis] - np.mean(z_pred[degree], axis=1, keepdims=True))**2 ) / nk
            variance[degree] += np.mean( np.var(z_pred[degree], axis=1, keepdims=True) ) / nk
    plt.plot(np.arange(n_degrees), error,    color=f"C{i}", linestyle="-", label=fr"$\lambda =$ {lam}")
    plt.plot(np.arange(n_degrees), bias,     color=f"C{i}", linestyle="--")
    plt.plot(np.arange(n_degrees), variance, color=f"C{i}", linestyle="-.")

plt.plot([], [], color="k", linestyle="-",  label="Error")
plt.plot([], [], color="k", linestyle="--", label="Bias")
plt.plot([], [], color="k", linestyle="-.", label="Variance")
plt.yscale('log')
plt.legend()
plt.show()