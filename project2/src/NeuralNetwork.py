import numpy as np
import jax
from tqdm import trange, tqdm

class DenseLayer:
    """
    Dense layer class to be used with NeuralNetwork class
    Used as both container for network parameters (weights and biases) and activation functions,
    and as calculator for forward sweeps, where layers are called upon to calculate each step
    """
    def __init__(self, n_nodes, n_prev, activation_function, activation_derivative, seed=0):
        """
        Inputs: # of nodes in this layer, # of nodes in previous layer, and activation function
        Initializes weights and biases
        """
        self.n_nodes = n_nodes               # Saving number of nodes for easy access
        self.W = np.zeros((n_prev, n_nodes)) # Weights
        self.b = np.zeros(n_nodes)           # Biases
        self.act = activation_function       # Activation function
        self.dact = activation_derivative    # Derivative of activation function
        self.seed = seed
        self.init_wb()

    def __call__(self, x):
        # Forward sweep portion for this layer
        # Takes input from previous layer, transforms it with fully connected weights,
        # adds biases, then passes it through the activation function
        z = x @ self.W + self.b
        a = self.act(z)
        return z, a

    def init_wb(self):
        # Initialize weights and biases
        # Need to make sure parameters are non-zero to avoid gradient explosion, and
        # different initial values to make sure nodes diverge
        # Weights are most important due to their quantity compared to biases
        RNG = np.random.default_rng(seed = self.seed) # Try to get variation in layers random distributions
        self.W += RNG.normal(0, 1, size=self.W.shape)
        self.b += RNG.normal(0, 1, size=self.b.shape)

    def update_wb(self, weights, biases):
        # Take new updated weights and biases
        self.W = weights
        self.b = biases

    def reset_wb(self):
        # Resets weights and biases to initial state, useful for comparing outcome
        # with different setups of hyperparameters
        self.W *= 0
        self.b *= 0
        self.init_wb()

class NeuralNetwork:
    def __init__(self, input_nodes):
        """
        Take input parameters and set up necessary containters and values
        """
        self.input_nodes = input_nodes # Number of input nodes
        self.layers = []               # Container for all layers in network
        self.z = [0]                   # Pre- activation function values
        self.a = [0]                   # Post-activation function values
        self.i = 0                     # Number of times trained, used for gradient accumulation un-bias-ing in Adam
        #self.lam # Regularization term

    def __call__(self, x):
        """
        Calculate output for current weights/biases, useful for when network is done developing
        """
        self.FeedForward(x) # Do FF with input
        return self.a[-1]   # Activated output from final layer

    def set_CostFunc(self, costfunc=None, costgrad=None):
        """
        Sets cost function used to evaluate and update weights & biases
        Can be either string naming built-in method, or callable method.
        Analytical expression for cost gradient is used for built in methods, and
        can also be supplied through input as secondary method. If no gradient method
        is given, an automatic gradient is calculated using jax library.
        """
        if callable(costfunc): 
            self.costfunc = costfunc
            if costgrad == None:
                self.costgrad = jax.autograd(self.costfunc)
            elif callable(costgrad):
                self.costgrad = costgrad
            else:
                raise ValueError("Input for cost gradient is not valid, see documentation")
        elif isinstance(costfunc, str):
            if costfunc == "MSE": # Regression cost
                self.costfunc = lambda y, t: np.mean((t - y)**2)
                self.costgrad = lambda y, t: (y - t) * 2#/len(t)
            elif costfunc == "CrossEntropy": # Logistic cost
                self.costfunc = lambda y, t: np.sum(np.log(1 + np.exp(y)) - t*y)
                self.costgrad = lambda y, t: np.exp(y) / (1 + np.exp(y)) - t
            else: 
                raise ValueError(f"No cost function with identifier {costfunc} found, see documentation")
        else:
            raise ValueError("Input for cost function is not valid, see documentation")

    def set_LearningRate(self, eta=1e-3, method=None, delta=1e-7, rho1=0.9, rho2=0.999):
        """
        eta: Global/base learning rate
        method: Adaptive method
        Has to be called after layers are set, since weights/biases from layers define setup
        
        eta function inputs:
            l: layer l of L, used to know which gradient to update
            wb: if weights: wb=0, if biases: wb=1
            g: gradient (delta for biases, delta@a for weights)
            i: iteration number i, i.e. number of times eta has been called (starting at 1)
        eta function returns eta*grad

        all eta functions take same input to have consistent call structure
        """
        eta0 = eta
        if method == None:
            eta = lambda l, wb, g, i: eta0*g
        elif isinstance(method, str):
            self.r = [[np.zeros_like(l.W), np.zeros_like(l.b)] for l in self.layers] # Gradient accumulation variable
            self.s = [[np.zeros_like(l.W), np.zeros_like(l.b)] for l in self.layers] # Secondary accumulation variable
            if method == "AdaGrad":
                def eta(l, wb, g, i):
                    self.r[l][wb] += g**2 # Component-wise
                    return g*eta0/(delta + np.sqrt(self.r[l][wb]))
            elif method == "RMSProp":
                def eta(l, wb, g, i):
                    self.r[l][wb] = rho1*self.r[l][wb] + (1-rho1)*g**2
                    return g*eta0/np.sqrt(delta + self.r[l][wb])
            elif method == "Adam":
                def eta(l, wb, g, i):
                    # Algorithm 8.7 from https://www.deeplearningbook.org/contents/optimization.html, page 306
                    self.s[l][wb] = rho1*self.s[l][wb] + (1 - rho1)*g     # Update biased first moment estimate
                    self.r[l][wb] = rho2*self.r[l][wb] + (1 - rho2)*g**2  # Second moment estimate
                    return eta0*(self.s[l][wb]/(1-rho1**i))/(delta + np.sqrt(self.r[l][wb]/(1-rho2**i))) # Using unbiased estimates
            else:
                raise ValueError(f"No adaptive method named {method} found")
        else:
            raise ValueError(f"Adaptive method has to be either None or a valid string, see documentation")

        self.eta = eta

    def addLayer(self, type="Dense", n_nodes=100, act="Sigmoid", alpha=None, seed=0):
        """
        Adds later of given type and width to current network, in first to last order
        Alpha is a hyperparameter, set to default values if not specified
        """
        ### Either take in activation function, or set to any predefined types
        if callable(act):
             act = act
        elif act == "Sigmoid":
             act  = lambda z: 1/(1 + np.exp(-z))
             dact = lambda z: act(z)*(1 - act(z))
        elif act == "tanh":
             act  = lambda z: np.tanh(z)
             dact = lambda z: 1 - np.tanh(z)**2
        elif act == "ELU":
             alpha = 1.00 if not alpha else alpha
             act  = lambda z: np.where(z < 0, alpha*(np.exp(z) - 1), z)
             dact = lambda z: np.where(z < 0, alpha*np.exp(z), 1)
        elif act == "ReLU":
             act  = lambda z: np.where(z < 0, 0, z)
             dact = lambda z: np.where(z < 0, 0, 1)
        elif act == "LeakyReLU": # Actually Parametric ReLU with default state Leaky ReLU
             alpha = 0.01 if not alpha else alpha
             act  = lambda z: np.where(z < 0, alpha*z, z)
             dact = lambda z: np.where(z < 0, alpha, 1)
        elif act == "linear": # Output layer regression, aka Identity(z)
             act  = lambda z: z
             dact = lambda z: 1
        elif act == "Softmax": # Output layer multiple-category classification
             act  = lambda z: np.exp(z)/(np.sum(np.exp(z)))
             dact = jax.autograd(act)
        else:
            raise ValueError("No valid activation function callable or identifier input")

        ### If this is the initial hidden layer, weights are based on input nodes
        ### else, they are based on the previous layers node count
        if len(self.layers) == 0:
            prev_nodes = self.input_nodes
        else:
            prev_nodes = self.layers[-1].n_nodes

        ### Append layer of given type, only "Dense" layers are used in this project, but there is support for more
        if type == "Dense":
            self.layers.append(DenseLayer(n_nodes, prev_nodes, act, dact, seed))
        #elif type == ...
        else:
            raise ValueError("No valid layer type identifier input")

        self.z.append(0) # Add storage for output for new layer
        self.a.append(0)

    def FeedForward(self, x):
        """
        Do forward pass through network
        Calls upon layers in succession, calculation handled by individual layers
        Output from each layer is then the input to the next, with the final result being saved
        a and z differ in indexing/length because a also includes initial input x, which pushes the index forwards
        """
        self.a[0] = x
        for i, layer in enumerate(self.layers):
            zi, ai = layer(self.a[i])
            self.z[i  ] = zi
            self.a[i+1] = ai

    def BackPropogate(self, t):
        """
        Use last forward pass to update weights & biases through cost function
        Calculates gradient using cost function, and passes new weights and biases to layers
        """
        layers = self.layers
        z = self.z
        a = self.a
        eta = self.eta

        L = len(layers) - 1 # zero indexed
        delta = [0]*(L+1)

        ### Find errors in each layer delta, output layer is set separately
        delta[L] = layers[L].dact(z[L]) * self.costgrad(a[L+1], t) # delta_L = dsigma * dCda, a[L+1] due to x index push
        for l in range(L-1, -1, -1):
            delta[l] = np.matmul(layers[l+1].W, delta[l+1]) * layers[l].dact(z[l])

        ### Use errors in each layer to update parameters
        self.i += 1
        for l in range(L, -1, -1):
            layers[l].W -= eta(l, 0, np.outer(a[l], delta[l]), self.i)  # NB, a[l] not a[l-1] since inserted x pushes index
            layers[l].b -= eta(l, 1,                 delta[l], self.i)

    def train(self, X, t, epochs, seed=0, silent=False):
        """
        Do forward and backward sweeps for learning
        Does epochs*len(t) iterations, where X,t are shuffled for each epoch
        """
        RNG = np.random.default_rng(seed)
        if silent:
            epochs_r = range(epochs)
        else:
            epochs_r = trange(epochs)

        for epoch in epochs_r:
            p = RNG.permutation(len(t))
            X_, t_ = X[p], t[p]
            for i in range(len(t)):
                self.FeedForward(X_[i])
                self.BackPropogate(t_[i])

    def test(self, x):
        """
        Calls upon network for all points in x
        """
        a = np.zeros((len(x), self.layers[-1].n_nodes), dtype=float)
        for i, xi in enumerate(x):
            a[i] = self.__call__(xi)

        return a

    def classify(self, X):
        """
        Calls upon network for all points in x, but also uses binary classification,
        where 0 is returned if output(xi) in X is less than (or equal to) 0.5, or 1 if output is larger than 0.5

        Would add functionality for multiple category translation here in
        future, i.e. [0.2, 0.2, 0.5, 0.1] -> 3 (2 for zero indexing), currently assumes
        only one node (binary problem)

        Merge with 'test' function, adding 'classify' as keyword?
        """
        a = np.zeros((len(X)), dtype=float)
        for i, xi in enumerate(X):
            a[i] = 1 if self.__call__(xi) > 0.5 else 0

        return a

    def reset_parameters(self):
        """
        Re-initializes parameters, used when comparing outcomes for different hyperparameters
        Should return to exact same state as before, due to seeded randomness
        """
        for layer in self.layers:
            layer.reset_wb()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    seed = 1 # Used for initialization of parameters, and drawing of samples

    # 1D data
    x = np.linspace(0, 1, 101, dtype=float).reshape(-1, 1) # Column vector
    t = -1*x + 3*x**2 - 2*x**3
    t += np.random.normal(0, max(abs(t))*0.2, t.shape)

    NN = NeuralNetwork(input_nodes=1)
    NN.addLayer(n_nodes=100, seed=seed,   act="Sigmoid") # Hidden layer
    NN.addLayer(n_nodes=100, seed=seed+1, act="Sigmoid") # Hidden layer
    NN.addLayer(n_nodes=  1, seed=seed+2, act= "linear") # Output layer
    NN.set_LearningRate(eta=1e-4, method="Adam")
    NN.set_CostFunc("MSE")
    NN.train(x, t, epochs=1000, seed=seed)
    y = NN.test(x)

    plt.plot(x, t, label="Target")
    plt.plot(x, y, label="Model")
    plt.legend()
    plt.show()

    # Multiple learning rates
    etas = np.logspace(-4, -1, 20)
    epochs = 100
    epoch_step = 1
    MSE = np.zeros((len(etas), int(epochs/epoch_step)))
    NN = NeuralNetwork(input_nodes=1)
    NN.addLayer(n_nodes=100, seed=seed,   act="Sigmoid") # Hidden layer
    NN.addLayer(n_nodes=100, seed=seed+1, act="Sigmoid") # Hidden layer
    NN.addLayer(n_nodes=  1, seed=seed+2, act= "linear") # Output layer
    NN.set_CostFunc("MSE")

    from sklearn.model_selection import train_test_split
    x_l, x_t, t_l, t_t = train_test_split(x, t)

    for i, eta in tqdm(enumerate(etas)):
        NN.reset_parameters() # Re-initialize weights/biases for each eta
        NN.set_LearningRate(eta=eta, method=None)
        for j in range(int(epochs/epoch_step)):
            NN.train(x_l, t_l, epochs=epoch_step, seed=seed, silent=True)
            MSE[i][j] = np.mean((NN.test(x_t) - t_t)**2)

    MSE = np.nan_to_num(MSE, nan=np.nan, posinf=np.nan, neginf=np.nan)
    plt.contourf(np.log10(MSE), levels=25, extent=[0, epochs, np.log10(etas[0]), np.log10(etas[-1])])
    plt.colorbar(label=r"log$_{10}$(MSE)")
    plt.xlabel("# of Epochs")
    plt.ylabel(r"log$_{10}(\eta)$")
    plt.show()

    # 2D data
    def FrankeFunction(x,y):
        term1 =  0.75*np.exp(-(9*x-2)**2/4.00 - (9*y-2)**2/4.00)
        term2 =  0.75*np.exp(-(9*x+1)**2/49.0 - (9*y+1)   /10.0)
        term3 =  0.50*np.exp(-(9*x-7)**2/4.00 - (9*y-3)**2/4.00)
        term4 = -0.20*np.exp(-(9*x-4)**2      - (9*y-7)**2     )
        return term1 + term2 + term3 + term4

    nx, ny = 20, 20
    X,Y = np.meshgrid(np.linspace(0, 1, nx+1, dtype=float),
                      np.linspace(0, 1, ny+1, dtype=float))
    Z = FrankeFunction(X, Y)
    x = np.array(list(zip(X.flatten(), Y.flatten())))
    t = Z.flatten().reshape(-1, 1)

    NN = NeuralNetwork(input_nodes=2)
    NN.addLayer(n_nodes=1000, seed=seed, act="Sigmoid")
    NN.addLayer(n_nodes= 100, seed=seed, act="Sigmoid")
    NN.addLayer(n_nodes=   1, seed=seed, act= "linear")
    NN.set_LearningRate(eta=1e-2, method="AdaGrad")
    NN.set_CostFunc("MSE")
    NN.train(x, t, epochs=100, seed=seed)
    Z_model = NN.test(x).reshape(Z.shape)

    fig = plt.figure(figsize=plt.figaspect(1./2))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax0.plot_surface(X, Y, Z)
    surf = ax1.plot_surface(X, Y, Z_model)
    ax0.view_init(30, 90)
    ax1.view_init(30, 90)
    plt.show()
