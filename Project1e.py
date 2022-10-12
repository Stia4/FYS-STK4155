from Project1b import create_X, SVDinv, make_Franke
from Project1d import kfold
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import trange
plt.rcParams.update({'font.size':16})

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

from Project1b import make_Franke, train_test_data_evenly, MSE, R2_Score
from sklearn.metrics import mean_squared_error as MSE_skl
from sklearn.metrics import r2_score as R2_skl

def main_1e():
    ### Make goal data
    nx, ny = (30, 30)
    x, y, z = make_Franke(nx, ny, noise = 0.1, seed=0)

    ### Split data into training and test data, using ~25% as test data
    test_size = 4 # Reserves 1/4 of the data as test data
    x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst = train_test_data_evenly(x, y, z, test_size)

    orders = [0, 1, 2, 3, 4, 5]

    ### Defining input parameters
    nx, ny = (10, 10)                 # number of Franke points to generate in x,y
    n_degrees = 10                    # number of degrees, degrees 0 ... n_degrees-1
    orders = np.linspace(0, n_degrees-1, n_degrees)
    n_bootstraps = 100               # number of times to resample each learn-set
    lambdas = [1e-10, 1e-5, 1e-1] # values of lambda to use in Ridge regression
    lambdas_str = [r"10$^{-10}$", r"10$^{-5}$", r"10$^{-1}$"]

    ### Making dataset
    x, y, z = make_Franke(nx, ny, noise=0.1, seed=0)

    # Bootstrap
    test_size = 4
    x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst = train_test_data_evenly(x, y, z, test_size)

    ### Looping over lambdas, each generating a set of bias/variance/error curves
    for i, lam in enumerate(lambdas):
        bias     = np.zeros(n_degrees)
        variance = np.zeros(n_degrees)
        error    = np.zeros(n_degrees)

        # Performing n_bootstraps resamplings of learn set and resulting n_degrees models to each test set
        z_pred = Bootstrap_pred(x_lrn, x_tst, y_lrn, y_tst, z_lrn, z_tst, n_bootstraps, n_degrees, lam) # <- Note lambda is input here

        # Finding how well each model fits across all bootstraps, doing running average over all k-folds
        for degree in range(n_degrees): #         Broadcasting ↓   Axis with bootstrapped values ↓
            error[degree]    += np.mean( np.mean((z_tst[:, np.newaxis] - z_pred[degree])**2, axis=1, keepdims=True) )
            bias[degree]     += np.mean( (z_tst[:, np.newaxis] - np.mean(z_pred[degree], axis=1, keepdims=True))**2 )
            variance[degree] += np.mean( np.var(z_pred[degree], axis=1, keepdims=True) ) / nk

        ### Plotting curves for current lambda, same colour for same lambda but different linestyles
        plt.plot(np.arange(n_degrees), error,    color=f"C{i}", linestyle="-", label=r"$\lambda =$ {}".format(lambdas_str[i]))
        plt.plot(np.arange(n_degrees), bias,     color=f"C{i}", linestyle="--")
        plt.plot(np.arange(n_degrees), variance, color=f"C{i}", linestyle="-.")

    figname = lambda name: f"fig/{name}_Ridge_{min(orders)}-{max(orders)}_{nx}x{ny}.pdf"

    ### Formatting and showing plot
    plt.title("Bias-variance tradeoff for Ridge with Bootstrap")
    plt.xlabel("Polynomial order")
    plt.yscale('log')
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.legend(loc='best', bbox_to_anchor=(0.37, 0.22, 0., 0.3))
    plt.savefig(figname("bias-variance_tradeoff"), format="pdf")
    plt.show()

if __name__ == "__main__":
    main_1e()
