from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.metrics import mean_squared_error as MSE_skl
from sklearn.metrics import r2_score as R2_skl
from sklearn.linear_model import Lasso
plt.rcParams.update({'font.size':16})

def FrankeFunction(x,y):
	"""
	Base function to which we fit the polynomials
	"""
	term1 =  0.75*np.exp(-(9*x-2)**2/4.00 - (9*y-2)**2/4.00)
	term2 =  0.75*np.exp(-(9*x+1)**2/49.0 - (9*y+1)   /10.0)
	term3 =  0.50*np.exp(-(9*x-7)**2/4.00 - (9*y-3)**2/4.00)
	term4 = -0.20*np.exp(-(9*x-4)**2      - (9*y-7)**2     )
	return term1 + term2 + term3 + term4

def create_X(x, y, n = 5):
	"""
	Function for creating a X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.] (design matrix)
	Input is x and y mesh, or raveled mesh, keyword agruments n is the degree of the polynomial
	you want to fit.
	"""
	if len(x.shape) > 1: # Unravels/flattens multidimensional arrays
		x = x.flatten()
		y = y.flatten()

	N = len(x)
	l = int((n + 1)*(n + 2)/2) # Number of elements in beta
	X = np.ones((N, l))

	for i in range(1, n + 1):
		q = int((i)*(i + 1)/2)
		for k in range(i + 1):
			X[:,q + k] = (x**(i - k))*(y**k)

	return X

def train_test_data_evenly(x, y, z, n = 4):
	"""
	Takes in x, y, and z arrays, and splits each into learn and train sets.
	Puts every n-th element into test set and rest in training set.
	E.g. n = 4 sets 1/4 of the data as test and 3/4 as training.
	"""
	if len(x.shape) > 1:
		x = x.flatten()
		y = y.flatten()
		z = z.flatten()

	nth = slice(None, None, n)
	x_learn = np.delete(x, nth)
	y_learn = np.delete(y, nth)
	z_learn = np.delete(z, nth)
	x_test  = x[nth]
	y_test  = y[nth]
	z_test  = z[nth]

	return x_learn, y_learn, z_learn, x_test, y_test, z_test

def combine_train_test(z_learn, z_test, n = 4):
	"""
	Recombines test and train data which has been split by train_test_data_evenly.
	"""
	z = np.zeros(len(z_learn)+len(z_test))

	mask = np.zeros(z.shape, dtype=bool)
	mask[slice(None, None, n)] = True # test: true, train: false

	z[~mask] = z_learn
	z[ mask] = z_test

	return z

def MSE(y, y_tilde):
	"""
	Function for computing mean squared error.
	Input is y: analytical solution, y_tilde: computed solution.
	"""
	return np.sum((y-y_tilde)**2)/y.size

def R2_Score(y, y_tilde):
	"""
	Function for computing the R2 score.
	Input is y: analytical solution, y_tilde: computed solution.
	"""
	return 1 - np.sum((y[:-2]-y_tilde[:-2])**2)/np.sum((y[:-2]-np.average(y))**2)

def make_Franke(nx, ny, noise = 0.0, seed=None):
    """
    Creates Franke surface for given number of points in x and y.
    Optionally adds noise where the input is the fraction of the mean as standard deviation.
    """
    x = np.arange(0, 1, 1.0/nx)
    y = np.arange(0, 1, 1.0/ny)
    x_, y_ = np.meshgrid(x, y)

    z = FrankeFunction(x_, y_)
    z += np.random.default_rng(seed).normal(0, np.mean(z)*noise, z.shape) # Adding some normal noise
    z -= np.mean(z) # Centering the data
    return x_, y_, z

def SVDinv(A):
	"""
	Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
	"""
	U, s, VT = np.linalg.svd(A)

	#D = np.zeros((len(U),len(VT)))
	D = np.diag(s)

	UT = np.transpose(U)
	V = np.transpose(VT)
	invD = np.linalg.inv(D)

	return np.matmul(V,np.matmul(invD,UT))

def fit_OLS(orders, lam, x_lrn, y_lrn, z_lrn, x_tst=None, y_tst=None, z_tst=None):
    """
    Calculates best fit model for given data, using the input polynomial orders.
    Uses minimisation of Ordinary Least Squares to fit model, and Singular Value
    Decomposition to find inverse/pseudo-inverse.
    Allows either input for both learn and test set, or just learn set.
    """
    ### Calculate model fit using learn-set
    X_lrn = dict()
    beta = dict()
    z_lrn_model = dict()
    X_tst = dict()
    z_tst_model = dict()

    for i in orders:
        X_lrn[i] = create_X(x_lrn, y_lrn, i)
        beta[i] = SVDinv(X_lrn[i].T @ X_lrn[i]) @ X_lrn[i].T @ z_lrn
        z_lrn_model[i] = np.reshape(X_lrn[i] @ beta[i], z_lrn.shape)
        X_tst[i] = create_X(x_tst, y_tst, i)
        z_tst_model[i] = np.reshape(X_tst[i] @ beta[i], z_tst.shape)

    return X_lrn, X_tst, beta, z_lrn_model, z_tst_model

def fit_Ridge(orders, lam, x_lrn, y_lrn, z_lrn, x_tst=None, y_tst=None, z_tst=None):

    X_lrn = dict()
    beta = dict()
    z_lrn_model = dict()
    X_tst = dict()
    z_tst_model = dict()

    for i in orders:
        X_lrn[i] = create_X(x_lrn, y_lrn, i)
        I = np.identity(X_lrn[i].shape[1])
        beta[i] = SVDinv(X_lrn[i].T @ X_lrn[i] + lam*I) @ X_lrn[i].T @ z_lrn
        z_lrn_model[i] = np.reshape(X_lrn[i] @ beta[i], z_lrn.shape)
        X_tst[i] = create_X(x_tst, y_tst, i)
        z_tst_model[i] = np.reshape(X_tst[i] @ beta[i], z_tst.shape)

    return X_lrn, X_tst, beta, z_lrn_model, z_tst_model


def fit_Lasso(orders, lam, x_lrn, y_lrn, z_lrn, x_tst=None, y_tst=None, z_tst=None):
    X_lrn = dict()
    X_tst = dict()
    z_lrn_model = dict()
    z_tst_model = dict()
    beta = dict()
    lasso_model = Lasso(alpha=lam, max_iter=1000000, tol=1e-1)
    for degree in orders:
        X_tst[degree] = create_X(x_tst, y_tst, n=degree)
        X_lrn[degree] = create_X(x_lrn, y_lrn, n=degree)
        lasso_model.fit(X_lrn[degree], z_lrn)
        z_tst_model[degree] = lasso_model.predict(X_tst[degree])
        z_lrn_model[degree] = lasso_model.predict(X_lrn[degree])
        beta[degree] = lasso_model.coef_
    return X_lrn, X_tst, beta, z_lrn_model, z_tst_model


def Bootstrap(x, y, n_bootstraps, maxdegree, silent=False):
    """
    Code based on:
    https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#the-bias-variance-tradeoff
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    error = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)
    degrees = np.zeros(maxdegree)

    for degree in range(maxdegree):
        X_t = create_X(x_test[:, 0], x_test[:, 1], n=degree)

        y_pred = np.empty((y_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train, random_state = 0 + degree*n_bootstraps + i)
            y_pred[:, i] = X_t @ fit_OLS([degree], x_[:, 0], x_[:, 1], y_[:, 0])[1][degree]

        degrees[degree] = degree
        error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )

    return bias, variance, error, degrees

def Bootstrap_pred_Ridge(x_l, x_t, y_l, y_t, z_l, z_t, n_bootstraps, maxdegree, lam=0.0):
    """
    Modified bootstrap code to extract only prediction matrix
    """
    z_pred = np.empty((maxdegree, len(z_t), n_bootstraps))
    for degree in range(maxdegree):
        X_t = create_X(x_t, y_t, n=degree)
        for i in range(n_bootstraps):
            x_lr, y_lr, z_lr = resample(x_l, y_l, z_l, random_state = 0 + degree*n_bootstraps + i) # Manually set RNG to get different bootstraps, but consistent result
            z_pred[degree, :, i] = X_t @ fit_Ridge([degree], lam, x_lr, y_lr, z_lr)[1][degree]
    return z_pred

def Bootstrap_pred_Lasso(x_l, x_t, y_l, y_t, z_l, z_t, n_bootstraps, maxdegree, lam=0.0):
    """
    Modified bootstrap code to extract only prediction matrix
    """
    z_pred = np.empty((maxdegree, len(z_t), n_bootstraps))
    lasso_model = Lasso(alpha=lam)
    for degree in range(maxdegree):
        data_test = create_X(x_t, y_t, n=degree)
        for i in range(n_bootstraps):
            x_lr, y_lr, z_lr = resample(x_l, y_l, z_l, random_state = 0 + degree*n_bootstraps + i)
            data_learn = create_X(x_lr, y_lr, n=degree)
            z_pred[degree, :, i] = lasso_model.fit(data_learn, z_lr).predict(data_test)
    return z_pred


def kfold(x, y, z, k, seed=None):
    """
    Splits input data into learn (train) and test data for given k-fold k.
    Number of points in each x,y,z needs to be evenly divisible with k.
    Inputs x,y,z are flattened if they are multidimensional.
    Optional input for seed for shuffling, can also be set with np.random.seed(seed)
    """
    assert x.size == y.size and x.size == z.size, f"Inputs are not of equal size! Sizes x: {x.size}, y: {y.size}, z: {z.size}"
    assert k > 1 and type(k) == int, f"k has to be an integer larger than 1! You input k = {k}"
    assert z.size % k == 0, f"Number of elements in input does not evenly split into {k} sections! Elements in z: {z.size}"

    if len(z.shape) > 1:
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

    rng = np.random.default_rng(seed)
    x, y, z = rng.permutation([x, y, z], axis=1) # shuffle and copy/return

    ## Dimensions [x/y/z, fold number, data]
    n_t = z.size//k
    learn_set = np.zeros((3, k, z.size - n_t))
    test_set  = np.zeros((3, k, n_t))

    ## Get and store learn/test data
    for i, p in enumerate([x, y, z]):                    # [(0, x), (1, y), (2, z)]
        for j in range(k):                               # [0, 1, .., k-2, k-1]
            test_section = slice(j*n_t, (j+1)*n_t)       # Location of test data
            learn_set[i, j] = np.delete(p, test_section) # All except test data = Learn data
            test_set[i, j]  = p[test_section]            # Test data

    return learn_set, test_set
