from Project1b import create_X, SVDinv, make_Franke
from Project1d import kfold
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import trange

def fit_Ridge(orders, lam, x_lrn, y_lrn, z_lrn, x_tst=None, y_tst=None, z_tst=None):
    """
	Calculates best fit model for given data, using the input polynomial orders.
	Ridge regression
    """
    ### Calculate model fit using learn-set
    X_lrn = dict()
    beta_OLS = dict()
    z_lrn_model = dict()
    for i in orders:
        X_lrn[i] = create_X(x_lrn, y_lrn, i)
        I = np.identity(X_lrn[i].shape[1])
        beta_OLS[i] = SVDinv(X_lrn[i].T @ X_lrn[i] + lam*I) @ X_lrn[i].T @ z_lrn
        z_lrn_model[i] = np.reshape(X_lrn[i] @ beta_OLS[i], z_lrn.shape)
	
	### If given, apply model to test-set as well
    if (x_tst is not None and y_tst is not None and z_tst is not None):
        X_tst = dict()
        z_tst_model = dict()
        for i in orders:
            X_tst[i] = create_X(x_tst, y_tst, i)
            z_tst_model[i] = np.reshape(X_tst[i] @ beta_OLS[i], z_tst.shape)
        return X_lrn, X_tst, beta_OLS, z_lrn_model, z_tst_model

    ### If test-set not given, only return learn-set model results
    return X_lrn, beta_OLS, z_lrn_model

# def Bootstrap(x_l, x_t, y_l, y_t, z_l, z_t, n_bootstraps, maxdegree, lam=0.0, silent=False):
#     """
#     Code based on:
#     https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#the-bias-variance-tradeoff
#     """
#     error = np.zeros(maxdegree)
#     bias = np.zeros(maxdegree)
#     variance = np.zeros(maxdegree)

#     for degree in range(maxdegree):
#         X_t = create_X(x_t, y_t, n=degree)
#         z_pred = np.empty((len(z_t), n_bootstraps))
#         for i in range(n_bootstraps):
#             x_lr, y_lr, z_lr = resample(x_l, y_l, z_l, random_state = 0 + degree*n_bootstraps + i)
#             z_pred[:, i] = X_t @ fit_Ridge([degree], lam, x_lr, y_lr, z_lr)[1][degree]

#         error[degree] = np.mean( np.mean((z_t[:, np.newaxis] - z_pred)**2, axis=1, keepdims=True) )
#         bias[degree] = np.mean( (z_t - np.mean(z_pred, axis=1, keepdims=True))**2 )
#         variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) )

#         if not silent:
#             print('Polynomial degree:', degree)
#             print('Error:', error[degree])
#             print('Bias^2:', bias[degree])
#             print('Var:', variance[degree])
#             print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

#     return bias, variance, error

def Bootstrap_pred(x_l, x_t, y_l, y_t, z_l, z_t, n_bootstraps, maxdegree, lam=0.0):
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

def main_1e():
    ### Defining input parameters
    nx, ny = (10, 10)                 # number of Franke points to generate in x,y
    n_degrees = 20                    # number of degrees, degrees 0 ... n_degrees-1
    nk = 4                            # number of k-folds
    n_bootstraps = 100                # number of times to resample each learn-set
    lambdas = np.logspace(-10, -1, 5) # values of lambda to use in Ridge regression

    ### Making dataset
    x, y, z = make_Franke(nx, ny, noise=0.1, seed=0)
    learn_set, test_set = kfold(x, y, z, nk, seed=0)

    ### Looping over lambdas, each generating a set of bias/variance/error curves
    for i, lam in enumerate(lambdas):
        bias     = np.zeros(n_degrees)
        variance = np.zeros(n_degrees)
        error    = np.zeros(n_degrees)

        ### Looping over k-folds, using a different combination each time
        for k in trange(nk):
            x_lk, y_lk, z_lk = learn_set[:, k] # x,y,z for learn set nr. k
            x_tk, y_tk, z_tk =  test_set[:, k] # x,y,z for test set

            # Performing n_bootstraps resamplings of learn set and resulting n_degrees models to each test set
            z_pred = Bootstrap_pred(x_lk, x_tk, y_lk, y_tk, z_lk, z_tk, n_bootstraps, n_degrees, lam) # <- Note lambda is input here

            # Finding how well each model fits across all bootstraps, doing running average over all k-folds
            for degree in range(n_degrees): #         Broadcasting ↓   Axis with bootstrapped values ↓
                error[degree]    += np.mean( np.mean((z_tk[:, np.newaxis] - z_pred[degree])**2, axis=1, keepdims=True) ) / nk # <- Note division by nk to
                bias[degree]     += np.mean( (z_tk[:, np.newaxis] - np.mean(z_pred[degree], axis=1, keepdims=True))**2 ) / nk #    get a running average
                variance[degree] += np.mean( np.var(z_pred[degree], axis=1, keepdims=True) ) / nk

        ### Plotting curves for current lambda, same colour for same lambda but different linestyles
        plt.plot(np.arange(n_degrees), error,    color=f"C{i}", linestyle="-", label=fr"$\lambda =$ {lam}")
        plt.plot(np.arange(n_degrees), bias,     color=f"C{i}", linestyle="--")
        plt.plot(np.arange(n_degrees), variance, color=f"C{i}", linestyle="-.")

    ### Formatting and showing plot
    plt.plot([], [], color="k", linestyle="-",  label="Error") # Linestyle labels to distinguish error/bias/variance
    plt.plot([], [], color="k", linestyle="--", label="Bias")
    plt.plot([], [], color="k", linestyle="-.", label="Variance")
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main_1e()