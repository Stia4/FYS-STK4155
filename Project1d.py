from Project1b import make_Franke, fit_OLS_SVD, MSE
import numpy as np
import matplotlib.pyplot as plt

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
    print(x[:10])

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

### Setting parameters
nx, ny = (10, 10)
k = 10
orders = np.arange(13)

### Generating data and splitting into k folds of learn/test data
x, y, z = make_Franke(nx, ny, noise=0.1, seed=0)
lrn, tst = kfold(x, y, z, k, seed=0)

### Iterating over folds
running_MSE = np.zeros_like(orders, dtype=float)
for i in range(k):
    ## Fitting polynomials of all orders to fold i learn set, then applying to test data
    z_tst_model = fit_OLS_SVD(orders, lrn[0, i], lrn[1, i], lrn[2, i],
                                      tst[0, i], tst[1, i], tst[2, i])[-1]
    
    x_unshuffle = [np.where(x[0, :] == xi) for xi in tst[0, i]]
    y_unshuffle = [np.where(y[:, 0] == yi) for yi in tst[1, i]]
    z_data = z[x_unshuffle, y_unshuffle][:, 0, 0]

    for j in orders:
        running_MSE[j] += MSE(z_data, z_tst_model[j]) / k

plt.plot(orders, running_MSE)
plt.show()