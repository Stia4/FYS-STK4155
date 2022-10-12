from Project1b import make_Franke, train_test_data_evenly, fit_OLS, MSE, create_X
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from scipy import linalg

def Bootstrap(x, y, n_bootstraps, maxdegree, mindegree=0, step=1, silent=False):
    """
    Code based on:
    https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#the-bias-variance-tradeoff
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    degrees = np.arange(mindegree, maxdegree)[::step]
    
    error = np.zeros(len(degrees))
    bias = np.zeros(len(degrees))
    variance = np.zeros(len(degrees))
    n = len(y_train)
    
    for j in tqdm(range(len(degrees))):
        degree = degrees[j]
        
        X_t = create_X(x_test[:, 0], x_test[:, 1], n=degree)

        y_pred = np.empty((y_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train, random_state = 0 + degree*n_bootstraps + i)            
            y_pred[:, i] = X_t @ fit_OLS([degree], x_[:, 0], x_[:, 1], y_[:, 0])[1][degree]

        error[j] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        bias[j] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance[j] = np.mean( np.var(y_pred, axis=1, keepdims=True) )

        if not silent:
            print('Polynomial degree:', degree)
            print('Error:', error[j])
            print('Bias^2:', bias[j])
            print('Var:', variance[j])
            print('{} >= {} + {} = {}'.format(error[j], bias[j], variance[j], bias[j]+variance[j]))

    return bias, variance, error, degrees

def main_1c():
    ### Recreate Fig 2.11 of Hastie, Tibshirani, and Friedman
    ### Making coarse data with high order polynomial fits
    nx, ny = (10, 10)
    orders = np.arange(20)

    x, y, z = make_Franke(nx, ny, noise = 0.1, seed=0)
    x_l, y_l, z_l, x_t, y_t, z_t = train_test_data_evenly(x, y, z, n=4) # l = learn, t = test
    X_l, X_t, beta, z_lm, z_tm = fit_OLS(orders, x_l, y_l, z_l, x_t, y_t, z_t) # m = model

    figname = lambda name: f"fig/{name}_{min(orders)}-{max(orders)}_{len(x)}x{len(y)}.pdf"
    MSE_lrn = [MSE(z_l, z_lm[i]) for i in orders]
    MSE_tst = [MSE(z_t, z_tm[i]) for i in orders]
    plt.plot(orders, MSE_lrn, "o-", label="MSE Training data")
    plt.plot(orders, MSE_tst, "o-", label="MSE Test data")
    plt.hlines(0, min(orders), max(orders), colors="gray", linestyles="dashed")
    plt.xticks(ticks=orders)
    plt.xlabel("Polynome order")
    plt.ylabel("Mean squared error")
    plt.legend(loc=1)
    plt.grid(alpha = 0.3)
    plt.yscale('log')
    plt.tight_layout()
    #plt.savefig(figname("MSE_2.11"), format="pdf")
    plt.show()
    plt.clf()

    ### Bias-variance analysis
    #nx, ny = (10, 10)
    n_bootstraps = 100
    maxdegree = 10

    # Make data, transform into column vectors
    x, y, z = make_Franke(nx, ny, noise=0.1, seed=1)
    x = np.append(x.ravel().reshape(-1, 1), y.ravel().reshape(-1, 1), axis=1)
    y = z.ravel()
    y = y.reshape(y.size, 1)

    bias, variance, error, degrees = Bootstrap(x, y, n_bootstraps, maxdegree)

    plt.plot(degrees, error, label='Error')
    plt.plot(degrees, bias, label='bias')
    plt.plot(degrees, variance, label='Variance')
    plt.legend()
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    main_1c()
