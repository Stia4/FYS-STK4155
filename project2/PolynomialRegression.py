import numpy as np
import jax

def polyFit(order: int, x: np.ndarray, y: np.ndarray,
            costfunc = "OLS", lam = 0.0, beta: np.ndarray = None, 
            eta = 1e-3, adaptive_method: str = None, delta = 1e-7, rho = 0.9, rho2 = 0.999,
            autograd = False, momentum = 0.0,
            maxiter = 1e6, convergence_threshold = 1e-10, batch_size = 0, n_epochs = 0, min_grad=True,
            silent=False):
    """
    Fits a 1D function using a polynomial of given degree.
    Regression is done through minimization of cost functions "OLS" or "Ridge", with
    options for adaptive learning methods.

    --- Inputs ---

    Data and polynomial choice:
    - x, y : ndarray[N, ] of floats
        * Arrays containing variable x and and corresponding fit target y
    - order : int
        * Polynomial order to fit, e.g. order=2 tries to fit p(x) = a0 + a1*x + a2*x^2

    Cost function:
    - costfunc : str, default "OLS"
        * Either "OLS" or "Ridge" cost function which is minimized in regression process
    - lam : float >= 0, optional, default 0.0
        * Parameter for cost function "Ridge", should it be chosen. Does nothing for case "OLS".
    - autograd : bool, default False
        * Whether to use automatic differentiation of cost function for gradient calculation,
          if False analytical expression is used.

    Initial guess for beta:
    - beta : None or ndarray[order+1, ] of floats, default None
        * Initial guess for beta before iteration, if None it is set to zeros

    Learning rate:
    - eta : float > 0, default 1e-3
        * Gradient learning rate, may be modified through adaptive methods
    - adaptive_method : None or str, default None
        * Identifier for adaptive learning rates, either "AdaGrad", "RMSProp", or "Adam"
        * If left as none, the learning rate is constant at eta
    - delta : float > 0, default 1e-7
        * Parameter for learning rate methods, small value intended to avoid division by zero
    - rho : float ∈ (0, 1), default 0.9
        * Decay rate used in "RMSProp" and "Adam" methods, used as first estimator rho1 in "Adam"
    - rho2 : float ∈ (0, 1), default 0.999
        * Secondary decay rate for "Adam method"

    Optional momentum:
    - momentum : float ∈ [0, 1], default 0.0
        * Momentum parameter which for values > 0 adds momentum into gradient descent method

    Stopping criteria:
    - maxiter : int or float > 0, default 1e6
        * Maximal number of iterations for beta gradient descent, overwritten if n_epochs is used.
    - convergence_threshold : float > 0, default 1e-10
        * Threshold for cost function when beta is considered to have converged, only used if n_epochs is not set.
    - batch_size : int >= 0, default 0
        * If non-zero, instead of using all datapoints for gradient calculation, selects a random batch of given size
          per iteration to calculate gradient.
        * If non-zero, has to be less than, and evenly divisible into # of datapoints
        * Can be used without n_epochs, though convergence then is calculated for individual batches, which might be imprecise.
    - n_epochs : int >= 0, default 0
        * For batch size n and dataset size N, do n_epochs times N/n total iterations. Requires batch_size to be set.
        * Intended as free parameter to extend regression time, equivalent to requiring maxiter = n_epochs*N/n iterations.
    - min_grad : bool, default True
        * If True, exits early if gradient becomes unchanging over 1 iteration, sign that local/global minimum might be reached
        * Limited by numerical precision in comparison with previous and current gradient, also error prone if batch_size is
          used, as the gradient comparison requires fewer parameters to stagnate

    Misc:
    - silent : bool, default False
        * If True, do not print any information to the console, includes mid-iteration progress and exit reason

    --- Outputs ---

    - X : ndarray[N, order+1, ] of floats
        * Design matrix containing polynomial features for input data points
        * Ordered such that index X[i, j] contains x[i]**j
    - beta : ndarray[order+1, ]
        * Coefficients for polynomial features, found through regression
        * Coefficient beta[j] corresponds to feature x**j, for all x

    Model can be extracted through matrix multiplication as y_model = X @ beta
    """

    ### Setup design matrix with points and polynomial features
    X = np.empty((len(x), order+1)) # [points, features]
    for j in range(order+1):        # order+1 to include x**0 and x**order
        X[:, j] = x**j
    n_points = y.size

    ### Create cost function and derivative, either OLS or Ridge
    ### NB! If batch size is set, cost is calculated only for batch, not entire set
    if costfunc == "OLS":
        Cost = lambda beta: np.mean((X @ beta - y)**2) # Cost function
        dCdb = lambda beta: 2/n_points * X.T @ (X @ beta - y) # Derivative of cost function by beta
    elif costfunc == "Ridge":
        Cost = lambda beta: np.mean((X @ beta - y)**2) + lam*np.mean(beta**2)
        dCdb = lambda beta: 2/n_points * X.T @ (X @ beta - y) + lam * beta
        if lam == 0:
            print("Hyperparameter lambda is zero, Ridge method is equal to OLS")
    else:
        raise ValueError(f"Regression method has to be either 'OLS' or 'Ridge', you input: {costfunc}")

    ### Input option processing, setting initial beta
    if type(beta) == type(None): # Set initial guess for beta to zero if not given
        beta = np.zeros(order+1)
    else: # If beta is given, check that it is correct format
        assert len(beta.shape) == 1 and len(beta) == order+1, "Input beta has to be 1D array of length order + 1"

    ### Choose either automatic or analytical differentiation
    if autograd:
        grad_func = jax.grad(Cost)
    else:
        grad_func = dCdb

    ### Set up batch size system
    RNG = np.random.default_rng(seed=82435) # PCG64
    p = RNG.permutation(len(x))             # Shuffling permutation, used to restore order later
    X_ = X[p]; y_ = y[p]                    # Shuffling data to avoid batch patterns
    if not batch_size: # If no batch size is chosen, simply use all the data
        data = lambda: (X_, y_)
    else:              # If batch size is chosen, return random batch_size points
        assert n_points % batch_size == 0, f"Data size {n_points} does not divide evenly into batch size {batch_size}!"
        M = int(n_points/batch_size)
        def data():
            m = batch_size*RNG.integers(0, M) # Draw random integer in [0, M), then scale it with batch_size to get step size M
            return X_[m:m+M], y_[m:m+M]

    ### Start iterating on beta until max iterations is hit or it converges within threshold
    i = 1
    eps = Cost(beta)
    v = 0
    if min_grad:
        grad_prev = np.zeros_like(grad_func(beta)) # Gradient check, exits early if gradient does not change

    ### Set learning rate to either a constant or valid implemented method
    eta0 = eta
    if adaptive_method == None: # If no adaptive method is set, or epoch method not used, do constant learning rate
        eta = lambda g, i: g*eta0
    elif callable(adaptive_method):
        eta = eta
    elif isinstance(adaptive_method, str):
        global r #NB! Replace with self calls instead for class implementation
        if adaptive_method == "AdaGrad":
            r = np.zeros_like(beta) # Gradient accumulation variable
            def eta(g, i):
                global r
                r += g**2 # Component-wise
                return g*eta0/(delta + np.sqrt(r))
        elif adaptive_method == "RMSProp":
            #global r
            r = np.zeros_like(beta) # Gradient accumulation variable
            def eta(g, i):
                global r
                r = rho*r + (1-rho)*g**2
                return g*eta0/np.sqrt(delta + r)
        elif adaptive_method == "Adam":
            global s # Secondary gradient estimator
            r = np.zeros_like(beta)
            s = np.zeros_like(beta)
            def eta(g, i):
                # Algorithm 8.7 from https://www.deeplearningbook.org/contents/optimization.html, page 306
                global r, s
                s =  rho*s + (1 - rho )*g    # Update biased first moment estimate
                r = rho2*r + (1 - rho2)*g**2 # Second moment estimate
                return eta0*(s/(1 -  rho**i))/(delta + np.sqrt(r/(1 - rho2**i))) # Using unbiased estimates
        else:
            raise ValueError(f"No adaptive method named {adaptive_method} found")
    else:
        raise ValueError(f"Adaptive method has to be either None or a valid string, see documentation")

    ### Choose loop condition, either batch size and epochs, or set number of max iterations and convergence
    if n_epochs and batch_size: # Both n_epochs and batch_size have to be set to use this condition
        maxiter = int(n_epochs * n_points/batch_size)
        condition = lambda: i <= maxiter
    else:                       # Else do maxiter and convergence, note batches can still be used here
        condition = lambda: i <= maxiter and eps > convergence_threshold

    ### Do main loop, inside a 'try' block so it can be stopped at will by user CTRL+C
    try:
        while condition():
            grad = grad_func(beta)           # Find gradient using set function
            v = momentum*v + eta(grad, i)    # Keep momentum if set, and get learning rate (eta*grad included in eta)
            beta = beta - v                  # Improve guess using gradient
            
            X, y = data()                    # Update data, either with all points or current batch (used in cost/gradient)

            i += 1                           # Count iteration
            eps = Cost(beta)                 # Calculate cost for use in convergence
            if i % 1000 == 0 and not silent: # Print progress to terminal every 1000 steps
                print(i, eps, end="                         \r")

            if min_grad:
                if (grad == grad_prev).all(): # If all gradient components are unchanging, we've reached a minimum and can exit early
                    if n_epochs and not silent: # Differing text if convergence threshold is relevant or not
                        print(f"Minimum reached with unchanging gradient, final cost {eps}, exiting after {i} of {maxiter} iterations.")
                    elif not silent:
                        print(f"Minimum reached with unchanging gradient, final cost {eps} of {convergence_threshold}, exiting after {i} of {maxiter} iterations.")
                    break
                grad_prev = grad
    except KeyboardInterrupt:
        if not silent:
            print(f"\nLoop exited by user, final cost {eps} of {convergence_threshold}, exiting after {i} of {maxiter} iterations...")
        pass

    ### Print why program stopped to console (if not set to silent)
    if not silent:
        if i >= maxiter and n_epochs:
            print(f"All epochs done, total {maxiter} iterations, final cost {eps}.")
        elif i >= maxiter and not n_epochs:
            print(f"Maximum number of iterations {maxiter} reached, final cost {eps} of threshold {convergence_threshold}")
        elif eps <= convergence_threshold and not n_epochs:
            print(f"Final convergence threshold {convergence_threshold} met, exited after {i} of {maxiter} iterations")

    ### Return complete, unshuffled data, and coefficients
    X = X_[np.argsort(p)]
    return X, beta


if __name__ == "__main__": # If program is run directly (as opposed to being imported)
    import matplotlib.pyplot as plt
    import seaborn as sns # Using seaborn for easy heatmap plotting
    plt.rcParams.update({'font.size': 14})

    ###########################
    ### Simple function fit ###
    ###########################

    def make_simple_function(a0, a1, a2):
        simple_function = lambda x: a0 + a1*x + a2*x*x
        return simple_function

    # Making data, simple 1D 2nd order polynomial
    x = np.linspace(0, 1, 30)
    beta = [0, -1, 2]
    f = make_simple_function(*beta)
    t = f(x)
    
    # Adding some slight noise to the data
    RNG = np.random.default_rng(seed=0)
    t += RNG.normal(0, max(abs(t))*0.05, size=len(t))

    # Using params dictionary to easily modify single input parameters later
    params = {'order': 2, 'x': x, 'y': t,
              'costfunc': "OLS", 'lam': 1e-15,
              'maxiter': 1e6, 'convergence_threshold': 1e-10,
              'beta': None, 'eta': 1e-2, 
              'adaptive_method': None, 'delta': 1e-8, 'rho': 0.9, 'rho2': 0.999,
              'autograd': False,
              'momentum': 0.0,
              'batch_size': 0, 'n_epochs': 0, 'min_grad': True,
              'silent': False}
    X, beta_model = polyFit(**params)

    print("Real function coefficients:", beta)
    print("Model function coefficients:", beta_model)

    plt.plot(x, t, ".k", label="Data")
    plt.plot(x, X @ beta_model, label=f"Model, deg={params['order']:d}")
    plt.title("Regression of polynomial features to data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("figures/simpleGD.pdf", format="pdf")
    plt.show()

    #############################
    ### Coefficient evolution ###
    #############################

    params['maxiter'] = 10     # Number of steps between each time we save beta
    params['silent'] = True    # Avoid filling console with text
    params['min_grad'] = False # Do all iterations, no early exit

    ### Tracking changes in coefficients
    max_maxiters = int(5e3) # Number of betas to save
    beta_list = np.zeros((2, max_maxiters, params['order']+1))
    for i in range(max_maxiters):
        X, beta_model = polyFit(**params) # Do fit for a few steps
        beta_list[0, i, :] = beta_model     # Then save current beta
        params['beta'] = beta_list[0, i, :] # And use it as a start for the next cycle

    ### Doing the same again with added momentum
    params['momentum'] = 0.80
    params['beta'] = None
    for i in range(max_maxiters):
        X, beta_model = polyFit(**params)
        beta_list[1, i, :] = beta_model
        params['beta'] = beta_list[1, i, :]

    ### Plotting
    fig, ax = plt.subplots(1, params['order']+1)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    iters = np.arange(max_maxiters) * params['maxiter']
    for i in range(params['order']+1):
        ax[i].plot(iters, beta_list[0, :, i], color="C0", label="GD")
        ax[i].plot(iters, beta_list[1, :, i], color="C1", label="GD w/80% momentum")
        try:
            ax[i].hlines(beta[i], xmin=min(iters), xmax=max(iters), colors="black", linestyles="--", label=r"$\beta$")
        except:
            pass
        ax[i].title.set_text(f"Coefficient {i}")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.supxlabel('# of iterations')
    fig.supylabel('Coefficient value')
    fig.legend(handles, labels, loc = (0.8, 0.2))
    fig.tight_layout()
    plt.savefig("figures/betaConvergence.pdf", format="pdf")
    plt.show()

    #############################################
    ### Learning rate / Ridge parameter space ###
    #############################################

    sns.set(font_scale=1.9)

    params['maxiter'] = 1000
    params['costfunc'] = 'Ridge'
    params['adaptive_method'] = "Adam"
    params['momentum'] = 0
    params['beta'] = None

    lambdas = np.logspace(-5, 1, 7)
    etas    = np.logspace(-5, 1, 7)
    
    accuracy = np.zeros((len(etas), len(lambdas)))
    Ridge = lambda X, beta, lam: np.mean((X @ beta - f(x))**2) + lam*np.mean(beta**2)

    for i, eta in enumerate(etas):
        for j, lam in enumerate(lambdas):
            params['eta'] = eta
            params['lam'] = lam
            X, beta_model = polyFit(**params)
            accuracy[i, j] = Ridge(X, beta_model, lam)

    accuracy = np.nan_to_num(accuracy, nan=np.nan, posinf=np.nan, neginf=np.nan) # Seaborn dislikes infinities
    accuracy = np.log10(accuracy)                                                # More readable format

    fig, ax = plt.subplots(figsize = (12, 10))
    sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis", cbar_kws={'label':r'log$_{10}$(Cost)'})
    ax.set_title(f"Cost after {params['maxiter']} steps, {params['adaptive_method']} adaptive learning rate")
    ax.set_ylabel("Learning rate $\eta$")
    ax.set_yticklabels([r'$10^{'+f'{int(np.log10(eta))}'+r'}$' for eta in etas])
    ax.set_xlabel("Ridge parameter $\lambda$")
    ax.set_xticklabels([r'$10^{'+f'{int(np.log10(lam))}'+r'}$' for lam in lambdas])
    plt.savefig("figures/eta_lambda_heatmap.pdf", format="pdf")
    plt.show()

    ##########################
    ### Testing parameters ###
    ##########################

    from sklearn.model_selection import train_test_split

    def parameter_variation(x, t, params, param_name, param_vals):
        if param_vals[0] == None:
            param_vals[0] = "None"
            print(f"{param_name:^15} val", " ".join([f"{val:^8}" for val in param_vals]))
            param_vals[0] = None
        elif isinstance(param_vals[0], str):
            print(f"{param_name:^15} val", " ".join([f"{val:^8}" for val in param_vals]))
        else:
            print(f"{param_name:^15} val", " ".join([f"{val:.2e}" for val in param_vals]))
        print(f"{'':^15} MSE", end="", flush=True)

        for val in param_vals:
            params[param_name] = val
            _, beta = polyFit(**params)
            
            order = params["order"]
            X = np.empty((len(x), order+1)) # [points, features]
            for j in range(order+1):        # order+1 to include x**0 and x**order
                X[:, j] = x**j

            MSE = np.mean((X @ beta - t)**2)
            print(f" {MSE:.2e}", end="", flush=True)
        print(end="\n")

    x = np.linspace(0, 1, 100)
    beta = [0, 1, -2]
    f = make_simple_function(*beta)
    t = f(x)

    x_train, x_test, t_train, t_test = train_test_split(x, t, train_size=0.75, random_state=0)

    params = {'order': 2, 'x': x_train, 'y': t_train, # Using params dictionary to easily modify single input parameters later
              'costfunc': "OLS", 'lam': 0,
              'maxiter': 1e3, 'convergence_threshold': 1e-100,
              'beta': None, 'eta': 1e-4,
              'adaptive_method': None, 'delta': 1e-8, 'rho': 0.9, 'rho2': 0.999,
              'autograd': False,
              'momentum': 0.0,
              'batch_size': 0, 'n_epochs': 0, 'min_grad': False,
              'silent': True}

    np.seterr(all="ignore") # Lots of errors ahead, choose to ignore them

    print("\n--- Base parameters ---")
    for key in ['order', 'costfunc', 'lam', 'maxiter', 'eta',
                'adaptive_method', 'momentum', 'batch_size', 'n_epochs']:
        print(key+":", params[key])
    print(end="\n")

    ### Basis: GD w/OLS, maxiter determines end
    x = x_test; t = t_test; p = params
    parameter_variation(x, t, p.copy(),           "order",                      [1, 2, 3, 4, 5, 6])
    parameter_variation(x, t, p.copy(),         "maxiter",            [1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    parameter_variation(x, t, p.copy(),             "eta",       [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0])
    parameter_variation(x, t, p.copy(),        "momentum",         [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parameter_variation(x, t, p.copy(), "adaptive_method", [None, "AdaGrad", "RMSProp", "Adam"])

    ### Ridge:
    print("\nSwitching to Ridge cost function\n")
    p["costfunc"] = "Ridge"
    parameter_variation(x, t, p.copy(), "lam", [1e-15, 1e-10, 1e-5, 1e-0])
    print("\nSwitching back to OLS cost function")
    p["costfunc"] = "OLS"

    ### SGD
    p["n_epochs"] = 1e3
    p["batch_size"] = 15
    print("\nTesting stochastic gradient descent, new base parameters:")
    print("n_epochs:", p["n_epochs"])
    print("batch_size:", p["batch_size"], "\n")
    parameter_variation(x, t, p.copy(), "batch_size", [3, 5, 15, 25])
    parameter_variation(x, t, p.copy(),   "n_epochs", [1e1, 1e2, 1e3, 1e4])
    print(end="\n")