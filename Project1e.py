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

    # lambdas = np.logspace(-10, -1, 5) # values of lambda to use in Ridge regression
    # for lam in lambdas:
    #
    #     ### Make design matrix, calculate beta for OLS, and get model, orders 2 to 6 covered
    #     X_lrn, X_tst, beta_OLS, z_lrn_model, z_tst_model = fit_Ridge(orders, lam, x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst)
    #
    #     MSE_lrn = [     MSE(z_lrn, z_lrn_model[i]) for i in orders]
    #     MSE_tst = [     MSE(z_tst, z_tst_model[i]) for i in orders]
    #     R2_lrn  = [R2_Score(z_lrn, z_lrn_model[i]) for i in orders]
    #     R2_tst  = [R2_Score(z_tst, z_tst_model[i]) for i in orders]
    #     MSE_lrn_skl = [MSE_skl(z_lrn, z_lrn_model[i]) for i in orders]
    #     MSE_tst_skl = [MSE_skl(z_tst, z_tst_model[i]) for i in orders]
    #     R2_lrn_skl  = [ R2_skl(z_lrn, z_lrn_model[i]) for i in orders]
    #     R2_tst_skl  = [ R2_skl(z_tst, z_tst_model[i]) for i in orders]
    #
    #     print("MSE min for {}x{} points, Ridge: {}".format(nx, ny, min(MSE_tst)))
    #
    #     plt.rcParams["figure.figsize"] = (12, 8)
    #     plt.rcParams.update({'font.size': 16})
    #
    #     ### Setting up figure name formats, saves to directory 'fig/'
    #     # figname = lambda name: f"img_e/{name}_{min(orders)}-{max(orders)}_{len(x)}x{len(y)}.pdf"
    #     figname = lambda name: f"img_e/{name}_{min(orders)}-{max(orders)}_{len(x)}x{len(y)}.png"
    #
    #     ### Plotting the MSE/R2 as function of polynomial degree
    #     # plt.plot(orders, MSE_lrn, "o-", label="MSE Training data")
    #     # plt.plot(orders, MSE_tst, "o-", label="MSE Test data")
    #     plt.plot(orders, MSE_tst, "o-", label=r"$\lambda$={:.2e}".format(lam))
    #     #plt.hlines(min(MSE_tst), min(orders), max(orders), colors="gray", linestyles="dashed",label="$MSE_{min}$")
    #     #plt.hlines(0, min(orders), max(orders), colors="blue", linestyles="dashed")
    #     plt.xticks(ticks=orders)
    #     plt.xlabel("Polynomial order")
    #     plt.ylabel("Mean squared error")
    #     plt.title('MSE for Ridge, test data')
    #     plt.legend(loc=1)
    #     plt.grid(alpha = 0.3)
    #     plt.tight_layout()
    #     # plt.savefig(figname("MSE_Ridge"), format="pdf")
    #     plt.savefig(figname("MSE_Ridge"), format="png")
    #     #ßplt.clf()
    #     # plt.show()
    #
    #     # plt.plot(orders, R2_lrn, "o-", label="R2 Score Training data")
    #     # plt.plot(orders, R2_tst, "o-", label="R2 Score Test data")
    #     # #plt.hlines(1, min(orders), max(orders), colors="gray", linestyles="dashed")
    #     # #plt.hlines(max(R2_tst), min(orders), max(orders), colors="gray", linestyles="dashed")
    #     # plt.xticks(ticks=orders)
    #     # plt.xlabel("Polynomial order")
    #     # plt.ylabel("R2 Score")
    #     # plt.title('R2 score for Ridge')
    #     # plt.legend(loc=4)
    #     # plt.tight_layout()
    #     # plt.grid(alpha = 0.3)
    #     # plt.savefig(figname("R2_Ridge"), format="pdf")
    #     # plt.savefig(figname("R2_Ridge"), format="png")
    #     # # plt.show()
    #
    #     # ### Coefficient comparison with errorbars?
    #     # err = dict()
    #     # var_noise = dict()
    #     # varb = dict()
    #     # for i in orders:
    #     # 	err[i] = z_lrn - z_lrn_model[i] # epsilon
    #     # 	var_noise[i] = np.var(err[i]) # sigma^2
    #     # 	varb[i] = var_noise[i] * np.diag(np.linalg.pinv(X_lrn[i].T @ X_lrn[i])) # var(beta_hat_OLS)
    #     # 	plt.errorbar([j for j in range(len(beta_OLS[i]))], beta_OLS[i], yerr=varb[i], fmt="o", label=f"Order {i}", capsize=5.0)
    #     # plt.xlabel("Coefficient #")
    #     # plt.ylabel("Coefficient value")
    #     # plt.grid(alpha = 0.3)
    #     # plt.legend()
    #     # plt.savefig(figname("Errorbars_Ridge"), format="pdf")
    #     # plt.savefig(figname("Errorbars_Ridge"), format="png")
    #     # #plt.clf()
    #     # # plt.show()
    #
    # plt.show()

    ### Defining input parameters
    nx, ny = (10, 10)                 # number of Franke points to generate in x,y
    n_degrees = 10                    # number of degrees, degrees 0 ... n_degrees-1
    orders = np.linspace(0, n_degrees-1, n_degrees)
    nk = 4                            # number of k-folds
    n_bootstraps = 100               # number of times to resample each learn-set
    # lambdas = np.logspace(-10, -1, 3) # values of lambda to use in Ridge regression
    lambdas = [1e-10, 1e-5, 1e-1] # values of lambda to use in Ridge regression
    lambdas_str = [r"10$^{-10}$", r"10$^{-5}$", r"10$^{-1}$"]

    ### Making dataset
    x, y, z = make_Franke(nx, ny, noise=0.1, seed=0)

    # Bootstrap
    test_size = 4
    x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst = train_test_data_evenly(x, y, z, test_size)
    # learn_set, test_set = kfold(x, y, z, nk, seed=0)


    ### Looping over lambdas, each generating a set of bias/variance/error curves
    for i, lam in enumerate(lambdas):
        bias     = np.zeros(n_degrees)
        variance = np.zeros(n_degrees)
        error    = np.zeros(n_degrees)

        # ### Looping over k-folds, using a different combination each time
        # for k in trange(nk):
        #     x_lk, y_lk, z_lk = learn_set[:, k] # x,y,z for learn set nr. k
        #     x_tk, y_tk, z_tk =  test_set[:, k] # x,y,z for test set

        # Performing n_bootstraps resamplings of learn set and resulting n_degrees models to each test set
        # z_pred = Bootstrap_pred(x_lk, x_tk, y_lk, y_tk, z_lk, z_tk, n_bootstraps, n_degrees, lam) # <- Note lambda is input here
        z_pred = Bootstrap_pred_Ridge(x_lrn, x_tst, y_lrn, y_tst, z_lrn, z_tst, n_bootstraps, n_degrees, lam) # <- Note lambda is input here

        # Finding how well each model fits across all bootstraps, doing running average over all k-folds
        for degree in range(n_degrees): #         Broadcasting ↓   Axis with bootstrapped values ↓
            error[degree]    += np.mean( np.mean((z_tst[:, np.newaxis] - z_pred[degree])**2, axis=1, keepdims=True) )
            bias[degree]     += np.mean( (z_tst[:, np.newaxis] - np.mean(z_pred[degree], axis=1, keepdims=True))**2 )
            variance[degree] += np.mean( np.var(z_pred[degree], axis=1, keepdims=True) ) / nk

        # # Finding how well each model fits across all bootstraps, doing running average over all k-folds
        # for degree in range(n_degrees): #         Broadcasting ↓   Axis with bootstrapped values ↓
        #     error[degree]    += np.mean( np.mean((z_tk[:, np.newaxis] - z_pred[degree])**2, axis=1, keepdims=True) ) / nk # <- Note division by nk to
        #     bias[degree]     += np.mean( (z_tk[:, np.newaxis] - np.mean(z_pred[degree], axis=1, keepdims=True))**2 ) / nk #    get a running average
        #     variance[degree] += np.mean( np.var(z_pred[degree], axis=1, keepdims=True) ) / nk

        ### Plotting curves for current lambda, same colour for same lambda but different linestyles
        plt.plot(np.arange(n_degrees), error,    color=f"C{i}", linestyle="-", label=r"$\lambda =$ {}".format(lambdas_str[i]))
        plt.plot(np.arange(n_degrees), bias,     color=f"C{i}", linestyle="--")
        plt.plot(np.arange(n_degrees), variance, color=f"C{i}", linestyle="-.")

    figname = lambda name: f"img_e/{name}_Ridge_{min(orders)}-{max(orders)}_{nx}x{ny}.pdf"

    ### Formatting and showing plot
    # plt.plot([], [], color="k", linestyle="-",  label="Error") # Linestyle labels to distinguish error/bias/variance
    # plt.plot([], [], color="k", linestyle="--", label="Bias")
    # plt.plot([], [], color="k", linestyle="-.", label="Variance")
    plt.title("Bias-variance tradeoff for Ridge with Bootstrap")
    plt.xlabel("Polynomial order")
    # plt.ylabel("Mean squared error")
    plt.yscale('log')
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    # plt.legend(loc="center left")
    plt.legend(loc='best', bbox_to_anchor=(0.37, 0.22, 0., 0.3))
    plt.savefig(figname("bias-variance_tradeoff"), format="pdf")
    plt.show()

if __name__ == "__main__":
    main_1e()
