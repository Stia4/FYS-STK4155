import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
plt.rcParams.update({'font.size': 14})

np.random.seed(2022)

# Function for performing OLS regression with Scikit-Learn
def LinReg(x_train, x_test, y_train, y_test, degrees, n_bootstraps):
    y_pred = np.empty((len(degrees), x_test.shape[0], n_bootstraps)) # Array for storing predicitions
    for i in range(len(degrees)):
        # Create design matrix and set up estimator for OLS
        model = make_pipeline(PolynomialFeatures(degree=degrees[i]), LinearRegression(fit_intercept=False))
        for j in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train)             # Draw random samples from the data for each bootstrap iteration
            model.fit(x_, y_)                               # Fit model to training data
            y_pred[i, :, j] = model.predict(x_test).ravel() # Test model on test data and store predictions
    return y_pred


# Function for performing decision tree regression with Scikit-Learn
def TreeReg(x_train, x_test, y_train, y_test, depths, n_bootstraps):
    y_pred = np.empty((len(depths), x_test.shape[0], n_bootstraps)) # Array for storing predicitions
    for i in range(len(depths)):
        regr = DecisionTreeRegressor(max_depth=depths[i], random_state=42) # Estimator
        for j in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train)    # Draw random samples from the data for each bootstrap iteration
            regr.fit(x_, y_)                       # Fit model to training data
            y_pred[i, :, j] = regr.predict(x_test) # Test model on test data and store predictions
    return y_pred


# Function for performing regression with a feed-forward neural network using Scikit-Learn
# Here we also scan which learning rate, eta, and regularization parameter, lambda, give best result
def NNReg(x_train, x_test, y_train, y_test, hidden_nodes, n_bootstraps):
    y_train = y_train.reshape(-1) # Get shape expected by Scikit-Learn method
    y_pred = np.zeros((len(hidden_nodes), x_test.shape[0], n_bootstraps)) # Array for storing predicitions
    for i in range(len(hidden_nodes)):
        # Initial values for learning rate, regularization parameter and R2score
        # Used to find the best model to use for bias-variance tradeoff analysis
        eta_ = 0
        lambd_ = 0.1
        eta_vals = np.logspace(-3, 1, 5)
        lambd_vals = np.logspace(-3, 1, 5)
        score = 0
        # Go through values of eta for constant lambda to find the combination giving the best model
        for j, eta in enumerate(eta_vals):
            dnn_ = MLPRegressor(hidden_layer_sizes=hidden_nodes[i], activation="logistic", solver="adam",
            alpha=lambd_, learning_rate_init=eta, max_iter=10000, random_state=42) # Estimator
            dnn_.fit(x_train, y_train)
            score_new = dnn_.score(x_test, y_test)
            if score_new > score:
                score = score_new
                eta_ = eta

        # Set eta to 0.1 if the no models exceeded an R2 score of 0 for various eta and constant lambda=0.1
        eta_ = 0.1 if eta_ == 0 else eta_;
        # score_ = 0
        # Go through values of lambda for eta found above
        for k, lam in enumerate(lambd_vals):
                dnn_ = MLPRegressor(hidden_layer_sizes=hidden_nodes[i], activation="logistic", solver="adam",
                                    alpha=lam, learning_rate_init=eta_, max_iter=10000, random_state=42) # Estimator
                dnn_.fit(x_train, y_train)              # Fit model to training data
                score_new = dnn_.score(x_test, y_test)  # Get R2 score
                if score_new > score: # Update score and lambda if score is better than current score
                    # score_ = score_new
                    score = score_new
                    lambd_ = lam

        print(f"Best R2score: {score} for eta={eta_} and lambda={lambd_}")

        # Use eta and lambda that gave the best model to preform regression and store
        # predictions for further bias-variance tradeoff analysis
        dnn = MLPRegressor(hidden_layer_sizes=hidden_nodes[i], activation="logistic", solver="adam",
                       alpha=lambd_, learning_rate_init=eta_, max_iter=10000, random_state=42)
        for l in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train)
            dnn.fit(x_, y_)
            y_pred[i, :, l] = dnn.predict(x_test)
    return y_pred


# Function computing the MSE, bias and variance
def bve(x_train, x_test, y_train, y_test, comp_arr, n_bootstraps, method="OLS"):

    if method == "OLS":
        func = LinReg
    elif method == "Trees":
        func = TreeReg
    elif method == "NN":
        func = NNReg

    # Get predictions from selected method
    y_pred = func(x_train, x_test, y_train, y_test, comp_arr, n_bootstraps)
    n_ = y_pred.shape[0]
    # Arrays for storing MSE, bias and variance
    MSE = np.zeros(n_)
    bias = np.zeros(n_)
    variance = np.zeros(n_)
    for i in range(n_):
        MSE[i] = np.mean( np.mean((y_test - y_pred[i])**2, axis=1, keepdims=True) )
        bias[i] = np.mean( (y_test - np.mean(y_pred[i], axis=1, keepdims=True))**2 )
        variance[i] = np.mean( np.var(y_pred[i], axis=1, keepdims=True) )
    return comp_arr, MSE, bias, variance


# make function to create data
def f(x, noise=0.1):
	return np.sin(x)/x + np.cos(x) +  np.random.normal(0, noise, x.shape)


start = 0.1; stop = 10 # range of x

n_input = 20 # number of inputs of set 1
x = np.linspace(start, stop, n_input).reshape(-1, 1)
y = f(x) #np.sin(x) + np.random.normal(0, noise, x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# data set 2 with more points
n2_input = 50
x2 = np.linspace(start, stop, n2_input).reshape(-1, 1)
y2 = f(x2)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2)

# data set 3 with even more points
n3_input = 100
x3 = np.linspace(start, stop, n3_input).reshape(-1, 1)
y3 = f(x3)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.2)

# collect each data set into lists
n = [n_input, n2_input, n3_input]
x_tst = [x_test, x2_test, x3_test]
y_tst = [y_test, y2_test, y3_test]
x_trn = [x_train, x2_train, x3_train]
y_trn = [y_train, y2_train, y3_train]

n_bootstraps = 50 # number of bootstraps to use
max_deg = 16
degrees = np.arange(1, max_deg, 1) # range of polynomial degrees to analyze

max_depth = 15
depths = np.arange(1, max_depth, 1)

eta_vals = np.logspace(-3, 1, 5) # learning rates to scan
lambd_vals = np.logspace(-3, 1, 5) # regularization parameters
hidden_nodes = [[i] for i in range(2,5,1)] # hidden nodes to analyze


# Analyze OLS over polynomial degree for each dataset
fig, ax = plt.subplots(figsize=(12,4), ncols=3)
for i in range(len(x_tst)):
    comp_arr_, MSE, bias, variance = bve(x_trn[i], x_tst[i], y_trn[i], y_tst[i], degrees, n_bootstraps, method="OLS")

    ax[i].plot(comp_arr_, MSE, c="black", label="MSE")
    ax[i].plot(comp_arr_, bias, c="red", label="Bias")
    ax[i].plot(comp_arr_, variance, c="blue", label="Variance")
    ax[i].set_title(f"Using {n[i]} datapoints")
    ax[i].set_xlabel(f"Polynomial degree")
    ax[i].set_yscale("log")
    ax[i].grid()
ax[0].set_ylabel("Error using OLS")
ax[0].legend()
plt.tight_layout()
plt.savefig("fig/tradeoff_OLS.pdf")
plt.show()


# Analyze decision trees over maximum depth of the tree
fig, ax = plt.subplots(figsize=(12,4), ncols=3)
for i in range(len(x_tst)):
    comp_arr_, MSE, bias, variance = bve(x_trn[i], x_tst[i], y_trn[i], y_tst[i], depths, n_bootstraps, method="Trees")
    ax[i].plot(comp_arr_, MSE, c="black", label="MSE")
    ax[i].plot(comp_arr_, bias, c="red", label="Bias")
    ax[i].plot(comp_arr_, variance, c="blue", label="Variance")
    ax[i].set_title(f"Using {n[i]} datapoints")
    ax[i].set_xlabel(f"Max tree depth")
    ax[i].set_yscale("log")
    ax[i].grid()
ax[0].set_ylabel("Error using Trees")
ax[0].legend()
plt.tight_layout()
plt.savefig("fig/tradeoff_trees.pdf")
plt.show()


# Neural networks, here we also scan which hyperparametrs give best result
eta_vals = np.logspace(-3, 1, 5) # learning rates to scan
lambd_vals = np.logspace(-3, 1, 5) # regularization parameters
hidden_nodes = [[i] for i in range(2,20,1)] # hidden nodes to analyze error of

fig, ax = plt.subplots(figsize=(12,4), ncols=3)
for i in range(len(x_tst)):
    comp_arr_, MSE, bias, variance = bve(x_trn[i], x_tst[i], y_trn[i].reshape(-1), y_tst[i], hidden_nodes, n_bootstraps, method="NN")
    ax[i].plot(comp_arr_, MSE, c="black", label="MSE")
    ax[i].plot(comp_arr_, bias, c="red", label="Bias")
    ax[i].plot(comp_arr_, variance, c="blue", label="Variance")
    ax[i].set_title(f"Using {n[i]} datapoints")
    ax[i].set_xlabel(f"Hidden nodes")
    ax[i].set_yscale("log")
    ax[i].grid()
ax[0].set_ylabel("Error using FFNN")
ax[0].legend()
plt.tight_layout()
plt.savefig("fig/tradeoff_nn.pdf")
plt.show()
