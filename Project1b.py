from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

#
# TODO:
# Noise in data
# Split data into train/test
# Actually use MSE to gauge fit
# Nicer plots
#

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_X(x, y, n = 5):
	"""
	Function for creating a X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.] (design matrix)
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polinomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def MSE(y, y_tilde):
	"""
	Function for computing mean squared error.
	Input is y: analytical solution, y_tilde: computed solution.
	"""
	return np.sum((y-y_tilde)**2)/y.size

### Make goal data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y) # shape (20, 20)
# HERE: Add noise?
# HERE: Split into train / test, avoid potential overfitting

### Make design matrix, calculate beta for OLS, and get model, orders 2 to 5 covered
X        = dict()
beta_OLS = dict()
z_model  = dict()
for i in range(2, 5+1):
    X[i]        = create_X(x, y, i)
    beta_OLS[i] = np.linalg.inv(X[i].T.dot(X[i])).dot(X[i].T).dot(np.ravel(z))
    z_model[i]  = np.reshape(X[i] @ beta_OLS[i], z.shape)

# HERE: Use MSE/R2/other metric to gauge fit

### Make plots
# Make them all show at once rather than one by one?
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()
for i in range(2, 5+1):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z_model[i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
