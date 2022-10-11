from all_funcs import *
from sklearn.utils import resample
plt.rcParams.update({'font.size':16})

from Project1b import make_Franke
from Project1d import kfold
from Project1e import Bootstrap_pred_Ridge, fit_Ridge
from Project1f import Bootstrap_pred_Lasso, fit_Lasso
from tqdm import tqdm

### Defining input parameters
n_degrees = 20                     # number of degrees, degrees 0 ... n_degrees-1
nk = 4                             # number of k-folds
lambdas = np.logspace(-12, -1, 20) # values of lambda to use in Ridge regression

nx, ny = (20, 20)
x, y, z = make_Franke(nx, ny, noise=0.1)

### Making dataset
learn_set, test_set = kfold(x, y, z, nk, seed=0)

bias_ = []
variance_ = []
error_ = []

### Looping over lambdas, each generating a set of bias/variance/error curves
for i, lam in tqdm(enumerate(lambdas)):
    bias     = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)
    error    = np.zeros(n_degrees)

    ### Looping over k-folds, using a different combination each time
    for k in range(nk):
        x_lk, y_lk, z_lk = learn_set[:, k] # x,y,z for learn set nr. k
        x_tk, y_tk, z_tk =  test_set[:, k] # x,y,z for test set

        # Performing n_bootstraps resamplings of learn set and resulting n_degrees models to each test set
        # z_pred = fit_Ridge(np.arange(n_degrees), lam, x_lk, y_lk, z_lk, x_tk, y_tk, z_tk)[-1]
        # z_pred = np.array([z_pred[i] for i in range(n_degrees)])[:, :, np.newaxis]

        z_pred = fit_Lasso(np.arange(n_degrees), lam, x_lk, y_lk, z_lk, x_tk, y_tk, z_tk)
        z_pred = np.array([z_pred[i] for i in range(n_degrees)])[:, :, np.newaxis]

        # Finding how well each model fits across all bootstraps, doing running average over all k-folds
        for degree in range(n_degrees): #         Broadcasting ↓   Axis with bootstrapped values ↓
            error[degree]    += np.mean( np.mean((z_tk[:, np.newaxis] - z_pred[degree])**2, axis=1, keepdims=True) ) / nk # <- Note division by nk to
            bias[degree]     += np.mean( (z_tk[:, np.newaxis] - np.mean(z_pred[degree], axis=1, keepdims=True))**2 ) / nk #    get a running average
            variance[degree] += np.mean( np.var(z_pred[degree], axis=1, keepdims=True) ) / nk

    bias_.append(bias)
    variance_.append(variance)
    error_.append(error)


### error_ = np.array(error_)

degrees = np.arange(n_degrees)
extent = [degrees[0], degrees[-1], np.log10(lambdas[0]), np.log10(lambdas[-1])]

figname = lambda name: f"img_new/{name}_{0}-{n_degrees-1}_{nx}x{ny}_nlambdas_{len(lambdas)}.pdf"

plt.contourf(np.log10(error_), extent=extent, levels=30)
ix, iy = np.where(np.min(error_) == error_)
print("Lasso, CV, MSE min = {}".format(np.min(error_)))
# print("Lasso, CV, MSE min = {}".format(np.min(error_)))
plt.plot(degrees[iy], np.log10(lambdas[ix]), 'rx', label=r'pol. degree = %.i, $\lambda$ = %.2e' % (degrees[iy], lambdas[ix]))
plt.xticks(degrees[::2])
#plt.yticks(np.log10(lambdas))
plt.title(f"Lasso regression with Cross-validation")
plt.xlabel("Polynomial degree")
plt.ylabel(r"log$_{10}$($\lambda$)")
plt.legend()
plt.colorbar(label=r"log$_{10}$(MSE)")
plt.tight_layout()
figure = plt.gcf()
figure.set_size_inches(8, 6)
plt.savefig(figname("MSE_Franke_Lasso_CV"), format="pdf")
plt.show()

# bootstrap

from Project1b import make_Franke, create_X
from Project1c import Bootstrap
from Project1e import Bootstrap_pred_Ridge, fit_Ridge
from Project1f import Bootstrap_pred_Lasso, fit_Lasso
from sklearn.model_selection import train_test_split

### Defining input parameters
n_degrees = 20                     # number of degrees, degrees 0 ... n_degrees-1
n_bootstraps = 100                  # number of times to resample each learn-set
lambdas = np.logspace(-12, -1, 20)  # values of lambda to use in Ridge regression

nx, ny = (20, 20)
x, y, z = make_Franke(nx, ny, noise=0.1)

x = x.flatten()
y = y.flatten()
z = z.flatten()

### Making dataset
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, random_state=0)

bias_ = []
variance_ = []
error_ = []

### Looping over lambdas, each generating a set of bias/variance/error curves
for i, lam in tqdm(enumerate(lambdas)):
    bias     = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)
    error    = np.zeros(n_degrees)

    # Performing n_bootstraps resamplings of learn set and resulting n_degrees models to each test set
    z_pred = Bootstrap_pred_Lasso(x_train, x_test, y_train, y_test, z_train, z_test, n_bootstraps, n_degrees, lam) # <- Note lambda is input here
    #z_pred = method(np.arange(n_degrees), lam, x_train, y_train, z_train, x_test, y_test, z_test)
    #z_pred = np.array([z_pred[i] for i in range(n_degrees)])
    #z_pred = z_pred[:, :, np.newaxis]

    # Finding how well each model fits across all bootstraps, doing running average over all k-folds
    for degree in range(n_degrees): #         Broadcasting ↓   Axis with bootstrapped values ↓
        error[degree]    += np.mean( np.mean((z_test[:, np.newaxis] - z_pred[degree])**2, axis=1, keepdims=True) ) / nk # <- Note division by nk to
        bias[degree]     += np.mean( (z_test[:, np.newaxis] - np.mean(z_pred[degree], axis=1, keepdims=True))**2 ) / nk #    get a running average
        variance[degree] += np.mean( np.var(z_pred[degree], axis=1, keepdims=True) ) / nk

    bias_.append(bias)
    variance_.append(variance)
    error_.append(error)

error_ = np.array(error_)

degrees = np.arange(n_degrees)
extent = [degrees[0], degrees[-1], np.log10(lambdas[0]), np.log10(lambdas[-1])]

figname = lambda name: f"img_new/{name}_{0}-{n_degrees-1}_{nx}x{ny}_nlambdas_{len(lambdas)}.pdf"

plt.contourf(np.log10(error_), extent=extent, levels=30)
ix, iy = np.where(np.min(error_) == error_)
print("Lasso, bootstrap, MSE min = {}".format(np.min(error_)))
plt.plot(degrees[iy], np.log10(lambdas[ix]), 'rx', label=r'pol. degree = %.i, $\lambda$ = %.2e' % (degrees[iy], lambdas[ix]))
plt.xticks(degrees[::2])
#plt.yticks(np.log10(lambdas))
plt.title(f"Lasso regression with Bootstrap")
plt.xlabel("Polynomial degree")
plt.ylabel(r"log$_{10}$($\lambda$)")
plt.legend()
plt.colorbar(label=r"log$_{10}$(MSE)")
plt.tight_layout()
figure = plt.gcf()
figure.set_size_inches(8, 6)
plt.savefig(figname("MSE_Lasso_bootstrap"), format="pdf")
plt.show()









quit()




def plot_MSE_R2(fit_method):
    if fit_method==fit_OLS:
        method = "OLS"
    elif fit_method==fit_Ridge:
        method = "Ridge"
    else:
        method = "Lasso"

    ### Make goal data
    nx, ny = (30, 30)
    x, y, z = make_Franke(nx, ny, noise = 0.1, seed=0)

    ### Split data into training and test data, using ~25% as test data
    test_size = 4 # Reserves 1/4 of the data as test data
    x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst = train_test_data_evenly(x, y, z, test_size)

    ### Make design matrix, calculate beta for OLS, and get model, orders 2 to 6 covered
    orders = [0, 1, 2, 3, 4, 5]
    lam = 1e-5

    X_lrn, X_tst, beta, z_lrn_model, z_tst_model = fit_method(orders, lam, x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst)

    ### Check fit using MSE and R2, printing as a nice table, using both our and sklearn's MSE/R2 functions
    MSE_lrn = [     MSE(z_lrn, z_lrn_model[i]) for i in orders]
    MSE_tst = [     MSE(z_tst, z_tst_model[i]) for i in orders]
    R2_lrn  = [R2_Score(z_lrn, z_lrn_model[i]) for i in orders]
    R2_tst  = [R2_Score(z_tst, z_tst_model[i]) for i in orders]

    MSE_min = np.min(MSE_tst)
    idx_min = np.where(MSE_tst==MSE_min)
    print(f"{method}, bootstrap: MSE min={MSE_min}, polydeg={orders[idx_min[0][0]]}")
    R2_max = np.max(R2_tst)
    idx_max = np.where(R2_tst==R2_max)
    print(f"{method}, bootstrap: R2 max={R2_max}, polydeg={orders[idx_max[0][0]]}")

    # figname = lambda name: f"img_b/{name}_{min(orders)}-{max(orders)}_{len(x)}x{len(y)}{method}.pdf"
    figname = lambda name: f"img_new/{name}_{min(orders)}-{max(orders)}_{len(x)}x{len(y)}_{method}_lambda_{lam}.pdf"

    plt.plot(orders, MSE_lrn, "o-", label="MSE Training data")
    plt.plot(orders, MSE_tst, "o-", label="MSE Test data")
    plt.xticks(ticks=orders)
    plt.xlabel("Polynomial order")
    plt.ylabel("Mean squared error")
    plt.title(f'MSE for {method}')
    plt.legend(loc=1)
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(figname("MSE"), format="pdf")
    #ßplt.clf()
    plt.show()

    plt.plot(orders, R2_lrn, "o-", label="R2 Score Training data")
    plt.plot(orders, R2_tst, "o-", label="R2 Score Test data")
    #plt.hlines(1, min(orders), max(orders), colors="gray", linestyles="dashed")
    #plt.hlines(max(R2_tst), min(orders), max(orders), colors="gray", linestyles="dashed")
    plt.xticks(ticks=orders)
    plt.xlabel("Polynomial order")
    plt.ylabel("R2 Score")
    plt.title(f'R2 score for {method}')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.grid(alpha = 0.3)
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(figname("R2"), format="pdf")
    plt.show()

    if method=="OLS":
        ### Coefficient comparison with errorbars?
        err = dict()
        var_noise = dict()
        varb = dict()
        for i in orders:
            err[i] = z_lrn - z_lrn_model[i] # epsilon
            var_noise[i] = np.var(err[i]) # sigma^2
            varb[i] = var_noise[i] * np.diag(np.linalg.pinv(X_lrn[i].T @ X_lrn[i])) # var(beta_hat_OLS)
            plt.errorbar([j for j in range(len(beta[i]))], beta[i], yerr=varb[i], fmt="o", label=f"Order {i}", capsize=5.0)
        plt.title(rf"Coefficients $\beta$, {method}")
        plt.xlabel("Coefficient #")
        plt.ylabel("Coefficient value")
        plt.grid(alpha = 0.3)
        plt.legend()
        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        # plt.savefig(figname("Errorbars"), format="pdf")
        #plt.clf()
        plt.show()

# plot_MSE_R2(fit_Lasso)
# quit()

def plot_with_bs_or_cv(fit_method):

    if fit_method==fit_OLS:
        method = "OLS"
    elif fit_method==fit_Ridge:
        method = "Ridge"
    else:
        method = "Lasso"

    ### Make goal data
    nx, ny = (30, 30)
    x, y, z = make_Franke(nx, ny, noise = 0.1, seed=0)

    ### Split data into training and test data, using ~25% as test data
    test_size = 5 # Reserves 1/4 of the data as test data
    x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst = train_test_data_evenly(x, y, z, test_size)

    ### Make design matrix, calculate beta for OLS, and get model, orders 2 to 6 covered
    maxdegree = 10
    orders = np.linspace(0, maxdegree-1, maxdegree, dtype=int)
    lam = 1e-5

    n_bootstraps = 10

    MSE_bootstrap_lrn = np.zeros(maxdegree)
    MSE_bootstrap_tst = np.zeros(maxdegree)

    for degree in orders:
        MSE_temp_tst = 0
        MSE_temp_lrn = 0
        for i in range(n_bootstraps):
            x_lrn_r, y_lrn_r, z_lrn_r = resample(x_lrn, y_lrn, z_lrn, random_state = 0 + degree*n_bootstraps + i)
            z_lrn_model, z_tst_model = fit_method([degree], lam, x_lrn_r, y_lrn_r, z_lrn_r, x_tst, y_tst, z_tst)[-2:]
            MSE_temp_lrn += MSE(z_lrn, z_lrn_model[degree])
            MSE_temp_tst += MSE(z_tst, z_tst_model[degree])
        MSE_bootstrap_lrn[degree] = MSE_temp_lrn / n_bootstraps
        MSE_bootstrap_tst[degree] = MSE_temp_tst / n_bootstraps

    figname_pdf = lambda name: f"img_new/{name}_{method}_{min(orders)}-{max(orders)}_{nx}x{ny}_nbs_{n_bootstraps}_bootstrap.pdf"

    MSE_min = np.min(MSE_bootstrap_tst)
    idx_min = np.where(MSE_bootstrap_tst==MSE_min)
    print(f"{method}, bootstrap: MSE min={MSE_min}, polydeg={orders[idx_min[0][0]]}")

    plt.plot(orders, MSE_bootstrap_lrn, "o-", label="MSE Training data")
    plt.plot(orders, MSE_bootstrap_tst, "o-", label="MSE Test data")
    plt.xticks(ticks=orders)
    plt.xlabel("Polynomial order")
    plt.ylabel("Mean squared error")
    plt.title(f'MSE for {method}, Bootstrap')
    plt.legend(loc=1)
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    # plt.savefig(figname("MSE"), format="pdf")
    #ßplt.clf()
    plt.show()

    # quit()

    k_folds = 10
    train_data, test_data = kfold(x, y, z, k_folds, seed=0)

    MSE_cv_lrn = np.zeros(maxdegree)
    MSE_cv_tst = np.zeros(maxdegree)

    for degree in orders:
        MSE_temp_lrn = 0
        MSE_temp_tst = 0
        for k in range(k_folds):
            x_lrn, y_lrn, z_lrn = (train_data[0, k], train_data[1, k], train_data[2, k])
            x_tst, y_tst, z_tst = (test_data[0, k], test_data[1, k], test_data[2, k])
            z_lrn_model, z_tst_model = fit_method([degree], lam, x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst)[-2:]
            MSE_temp_lrn += MSE(z_lrn, z_lrn_model[degree])
            MSE_temp_tst += MSE(z_tst, z_tst_model[degree])
        MSE_cv_lrn[degree] = MSE_temp_lrn / k_folds
        MSE_cv_tst[degree] = MSE_temp_tst / k_folds

    figname_pdf = lambda name: f"img_new/{name}_{method}_{min(orders)}-{max(orders)}_{nx}x{ny}_nbs_{n_bootstraps}_bootstrap.pdf"

    MSE_min = np.min(MSE_cv_tst)
    idx_min = np.where(MSE_cv_tst==MSE_min)
    print(f"{method}, CV: MSE min={MSE_min}, polydeg={orders[idx_min[0][0]]}")

    plt.plot(orders, MSE_cv_lrn, "o-", label="MSE Training data")
    plt.plot(orders, MSE_cv_tst, "o-", label="MSE Test data")
    plt.xticks(ticks=orders)
    plt.xlabel("Polynomial order")
    plt.ylabel("Mean squared error")
    plt.title(f'MSE for {method}, Cross-validation')
    plt.legend(loc=1)
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    # plt.savefig(figname("MSE"), format="pdf")
    #ßplt.clf()
    plt.show()

plot_with_bs_or_cv(fit_OLS)




from sklearn.utils._testing import ignore_warnings # Ignoring convergence warning, else
from sklearn.exceptions import ConvergenceWarning  # the console will be filled with
@ignore_warnings(category=ConvergenceWarning)      # walls of text

def contourplots(fit_method):

    if fit_method==fit_Ridge:
        method = "Ridge"
        lambdas = np.logspace(-14, -1, 10)
    else:
        method = "Lasso"
        lambdas = np.logspace(-18, -1, 10)


    nx, ny = (20, 20)
    noise = 0.1
    x, y, z = make_Franke(nx, ny, noise, seed=0)

    # Bootstrap
    test_size = 4
    x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst = train_test_data_evenly(x, y, z, test_size)

    maxdegree = 15
    orders = np.linspace(0, maxdegree-1, maxdegree, dtype=int)
    n_bootstraps = 100

    MSE_bootstrap = np.zeros((len(lambdas), maxdegree))

    from tqdm import trange

    for i in trange(len(lambdas)):
        for degree in orders:
            MSE_temp = 0
            for k in range(n_bootstraps):
                x_lrn_r, y_lrn_r, z_lrn_r = resample(x_lrn, y_lrn, z_lrn, random_state = 0 + degree*n_bootstraps + k)
                z_tst_model = fit_method([degree], lambdas[i], x_lrn_r, y_lrn_r, z_lrn_r, x_tst, y_tst, z_tst)[4]
                MSE_temp += MSE(z_tst, z_tst_model[degree])
            MSE_bootstrap[i][degree] = MSE_temp / n_bootstraps

    # Create contourplots of MSE for range of polynomial orders and lambdas
    x_, y_ = np.meshgrid(orders, np.log10(lambdas))

    figname_pdf = lambda name: f"img_new/{name}_{method}_{min(orders)}-{max(orders)}_nlambdas_{len(lambdas)}_{min(lambdas)}_{max(lambdas)}_{nx}x{ny}_nbs_{n_bootstraps}_bootstrap_noise_{noise}.pdf"

    # figname_png = lambda name: f"img_new/{name}_{method}_{min(orders)}-{max(orders)}_nlambdas_{len(lambdas)}_{min(lambdas)}_{max(lambdas)}_{nx}x{ny}_nbs_{n_bootstraps}_bootstrap.png"

    MSE_min = np.min(MSE_bootstrap)
    idx_min = np.where(MSE_bootstrap==MSE_min)
    # print(idx_min)
    print(f"{method}, bootstrap: MSE min={MSE_min}, lambda={lambdas[idx_min[0][0]]}, polydeg={orders[idx_min[1][0]]}")

    plt.contourf(x_, y_, np.log10(MSE_bootstrap), levels=30)
    plt.plot(x_[idx_min[0],idx_min[1]], y_[idx_min[0],idx_min[1]], "rx", markersize=12, label=r"pol.degree={}, $\lambda$={:.1e}".format(orders[idx_min[1][0]], lambdas[idx_min[0][0]]))
    plt.colorbar(label=r"log$_{10}$(MSE)")
    plt.title(f"{method} regression with Bootstrap")
    plt.xlabel("Polynomial order")
    plt.ylabel(r"log$_{10}(\lambda)$")
    # plt.yscale("log")
    plt.legend()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(figname_pdf("MSE"), format="pdf")
    # plt.savefig(figname_png("MSE"), format="png")
    plt.show()

    # Cross validation
    # maxdegree = 20
    # orders = np.linspace(0, maxdegree-1, maxdegree, dtype=int)
    # lambdas = np.logspace(-10, -1, 20)

    k_folds = 10
    train_data, test_data = kfold(x, y, z, k_folds, seed=0)

    MSE_cv = np.zeros((len(lambdas), maxdegree))

    for i in trange(len(lambdas)):
        for degree in orders:
            MSE_temp = 0
            for k in range(k_folds):
                x_lrn, y_lrn, z_lrn = (train_data[0, k], train_data[1, k], train_data[2, k])
                x_tst, y_tst, z_tst = (test_data[0, k], test_data[1, k], test_data[2, k])
                z_tst_model = fit_method([degree], lambdas[i], x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst)[4]
                MSE_temp += MSE(z_tst, z_tst_model[degree])
            MSE_cv[i][degree] = MSE_temp / k_folds

    # Create contourplots of MSE for range of polynomial orders and lambdasx
    x_, y_ = np.meshgrid(orders, np.log10(lambdas))

    figname_pdf = lambda name: f"img_new/{name}_{method}_{min(orders)}-{max(orders)}_nlambdas_{len(lambdas)}_{min(lambdas)}_{max(lambdas)}_{nx}x{ny}_k_{k_folds}_CV_noise_{noise}.pdf"

    # figname_png = lambda name: f"img_new/{name}_{method}_{min(orders)}-{max(orders)}_nlambdas_{len(lambdas)}_{min(lambdas)}_{max(lambdas)}_{nx}x{ny}_k_{k_folds}_CV.png"

    MSE_min = np.min(MSE_cv)
    idx_min = np.where(MSE_cv==MSE_min)
    # print(idx_min)
    print(f"{method}, CV: MSE min={MSE_min}, lambda={lambdas[idx_min[0][0]]}, polydeg={orders[idx_min[1][0]]}")

    plt.contourf(x_, y_, np.log10(MSE_cv), levels=30)
    plt.plot(x_[idx_min[0],idx_min[1]], y_[idx_min[0],idx_min[1]], "rx", markersize=12, label=r"pol.degree={}, $\lambda$={:.1e}".format(orders[idx_min[1][0]], lambdas[idx_min[0][0]]))
    plt.colorbar(label=r"log$_{10}$(MSE)")
    # plt.colorbar(label="MSE")
    plt.title(f"{method} regression with cross validation")
    plt.xlabel("Polynomial order")
    plt.ylabel(r"log$_{10}(\lambda)$")
    # plt.yscale("log")
    plt.legend()
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig(figname_pdf("MSE"), format="pdf")
    # plt.savefig(figname_png("MSE"), format="png")
    plt.show()

contourplots(fit_Lasso)
