from Project1b import make_Franke, fit_OLS, MSE
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

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

def main_1d():
    ### Setting parameters
    nx, ny = (10, 10)
    k = 10
    orders = np.arange(30)

    ### Generating data and splitting into k folds of learn/test data
    x, y, z = make_Franke(nx, ny, noise=0.1, seed=0)
    lrn, tst = kfold(x, y, z, k, seed=0)

    ### Iterating over folds
    running_MSE_tst = np.zeros_like(orders, dtype=float)
    running_MSE_lrn = np.zeros_like(orders, dtype=float)
    for i in range(k):
        ## Fitting polynomials of all orders to fold i learn set, then applying to test data
        z_lrn_model, z_tst_model = fit_OLS(orders, lrn[0, i], lrn[1, i], lrn[2, i],
                                                   tst[0, i], tst[1, i], tst[2, i])[-2:]
        for j in orders:
            running_MSE_tst[j] += MSE(tst[2, i], z_tst_model[j]) / k
            running_MSE_lrn[j] += MSE(lrn[2, i], z_lrn_model[j]) / k

    print("MSE min for {}x{} points, CV: {}".format(nx, ny, min(running_MSE_tst)))

    figname = lambda name: f"fig/{name}_{min(orders)}-{max(orders)}_{nx}x{ny}.pdf"

    plt.plot(orders, running_MSE_lrn, "o-", label="MSE Training data")
    plt.plot(orders, running_MSE_tst, "o-", label="MSE Test data")
    plt.hlines(0, min(orders), max(orders), colors="gray", linestyles="dashed")
    plt.xticks(ticks=orders[::2])
    plt.xlabel("Polynomial order")
    plt.ylabel("Mean squared error")
    plt.title(f"MSE for OLS with k-fold cross validation, k = {k}")
    plt.grid(alpha = 0.3)
    plt.yscale('log')
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.tight_layout()
    plt.legend()
    plt.savefig(figname("MSE_CV"), format="pdf")
    plt.show()

if __name__ == "__main__":
	main_1d()
