from Project1b import create_X, make_Franke, train_test_data_evenly
from Project1d import kfold
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.linear_model import Lasso

from sklearn.utils._testing import ignore_warnings # Ignoring convergence warning, else
from sklearn.exceptions import ConvergenceWarning  # the console will be filled with
@ignore_warnings(category=ConvergenceWarning)      # walls of text

def fit_Lasso(degrees, lam, x_l, y_l, z_l, x_t, y_t, z_t):
    """
    Modified bootstrap code to extract only prediction matrix
    """
    z_pred = np.empty((len(degrees), len(z_t)))
    lasso_model = Lasso(alpha=lam)
    for degree in degrees:
        data_test = create_X(x_t, y_t, n=degree)
        data_learn = create_X(x_l, y_l, n=degree)
        z_pred[degree, :] = lasso_model.fit(data_learn, z_l).predict(data_test)
    return z_pred

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

def main_1f():
    nx, ny = (10, 10)                 # number of Franke points to generate in x,y
    n_degrees = 10                    # number of degrees, degrees 0 ... n_degrees-1
    orders = np.linspace(0, n_degrees-1, n_degrees)
    n_bootstraps = 100               # number of times to resample each learn-set
    lambdas = lambdas = [1e-10, 1e-5, 1e-1]  # values of lambda to use in Ridge regression
    lambdas_str = [r"10$^{-10}$", r"10$^{-5}$", r"10$^{-1}$"]

    x, y, z = make_Franke(nx, ny, noise=0.1, seed=0)

    # Bootstrap
    test_size = 4
    x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst = train_test_data_evenly(x, y, z, test_size)

    for i, lam in enumerate(lambdas):
        bias     = np.zeros(n_degrees)
        variance = np.zeros(n_degrees)
        error    = np.zeros(n_degrees)
        z_pred = Bootstrap_pred(x_lrn, x_tst, y_lrn, y_tst, z_lrn, z_tst, n_bootstraps, n_degrees, lam)
        for degree in range(n_degrees):
            error[degree]    += np.mean( np.mean((z_tst[:, np.newaxis] - z_pred[degree])**2, axis=1, keepdims=True) )
            bias[degree]     += np.mean( (z_tst[:, np.newaxis] - np.mean(z_pred[degree], axis=1, keepdims=True))**2 )
            variance[degree] += np.mean( np.var(z_pred[degree], axis=1, keepdims=True) ) / nk

        plt.plot(np.arange(n_degrees), error,    color=f"C{i}", linestyle="-", label=r"$\lambda =$ {}".format(lambdas_str[i]))
        plt.plot(np.arange(n_degrees), bias,     color=f"C{i}", linestyle="--")
        plt.plot(np.arange(n_degrees), variance, color=f"C{i}", linestyle="-.")

    figname = lambda name: f"fig/{name}_Lasso_{min(orders)}-{max(orders)}_{nx}x{ny}.pdf"

    plt.title("Bias-variance tradeoff for Lasso with Bootstrap")
    plt.xlabel("Polynomial order")
    plt.yscale('log')
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.legend(loc="center left")
    plt.savefig(figname("bias-variance_tradeoff"), format="pdf")
    plt.show()

if __name__ == "__main__":
    main_1f()
