from Project1b import create_X, make_Franke
from Project1d import kfold
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.linear_model import Lasso

from sklearn.utils._testing import ignore_warnings # Ignoring convergence warning, else
from sklearn.exceptions import ConvergenceWarning  # the console will be filled with
@ignore_warnings(category=ConvergenceWarning)      # walls of text
def Bootstrap_pred(x_l, x_t, y_l, y_t, z_l, z_t, n_bootstraps, maxdegree, lam=0.0):
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

### Same main part as from E, only function has changed
def main_1f():
    nx, ny = (10, 10)                 # number of Franke points to generate in x,y
    n_degrees = 20                    # number of degrees, degrees 0 ... n_degrees-1
    nk = 4                            # number of k-folds
    n_bootstraps = 100                # number of times to resample each learn-set
    lambdas = np.logspace(-5, -1, 5)  # values of lambda to use in Ridge regression

    x, y, z = make_Franke(nx, ny, noise=0.1, seed=0)
    learn_set, test_set = kfold(x, y, z, nk, seed=0)
    for i, lam in enumerate(lambdas):
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

if __name__ == "__main__":
    main_1f()