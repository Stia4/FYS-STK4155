import numpy as np
import jax
from tqdm import trange
import Layers

class NeuralNetwork:
    def __init__(self, input_nodes):
        """
        Take input parameters and set up necessary containters and values
        """
        self.input_nodes = input_nodes # Number of input nodes, can be multidimensional
        self.layers = []               # Container for all layers in network
        #self.lam # Regularization term

    def __call__(self, x):
        """
        Calculate output for current weights/biases, useful for when network is done developing
        """
        #
        # HERE: Normalize input to [0,1]? Else numbers may blow up later
        #
        self.FeedForward(x)      # Do FF with input
        return self.layers[-1].a # Activated output from final layer

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
        self.iters = 1 # Iteration count, used by Adam method
        if method == None:
            eta = lambda l, wb, g: eta0*g
        elif isinstance(method, str):
            self.r = [[np.zeros_like(l.W), np.zeros_like(l.b)] for l in self.layers] # Gradient accumulation variable
            self.s = [[np.zeros_like(l.W), np.zeros_like(l.b)] for l in self.layers] # Secondary accumulation variable
            if method == "AdaGrad":
                def eta(l, wb, g):
                    self.r[l][wb] += g**2 # Component-wise
                    return g*eta0/(delta + np.sqrt(self.r[l][wb]))
            elif method == "RMSProp":
                def eta(l, wb, g):
                    self.r[l][wb] = rho1*self.r[l][wb] + (1-rho1)*g**2
                    return g*eta0/np.sqrt(delta + self.r[l][wb])
            elif method == "Adam":
                def eta(l, wb, g):
                    # Algorithm 8.7 from https://www.deeplearningbook.org/contents/optimization.html, page 306
                    self.s[l][wb] = rho1*self.s[l][wb] + (1 - rho1)*g     # Update biased first moment estimate
                    self.r[l][wb] = rho2*self.r[l][wb] + (1 - rho2)*g**2  # Second moment estimate
                    return eta0*(self.s[l][wb]/(1-rho1**self.iters))/(delta + np.sqrt(self.r[l][wb]/(1-rho2**self.iters))) # Using unbiased estimates
            else:
                raise ValueError(f"No adaptive method named {method} found")
        else:
            raise ValueError(f"Adaptive method has to be either None or a valid string, see documentation")

        self.eta = eta # Store to be used in network

    def addLayer(self, type="Dense", params=100, act="Sigmoid", alpha=None, seed=0):
        """
        Adds later of given type and width to current network, in first to last order
        Alpha is a hyperparameter, set to default values if not specified
        Params is a set of parameters, unique to each type of layer:
            - Dense: params is integer depicting number of nodes
            - Convolutional: params is [kernel_extent, n_filters, stride, padding], all integers
            - Pooling: params is [kernel_extent, pool_type], where kernel_extent is an integer and pool_type is a string
        See documentation for each separate layer to see further description of parameters
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
            prev_nodes = self.layers[-1].output_shape

        ### Append layer of given type
        if type == "Dense":
            if np.ndim(prev_nodes) != 0: # If previous layer output is multidimensional, flatten it
               prev_nodes = np.prod(prev_nodes)
            self.layers.append(Layers.DenseLayer(prev_nodes, params, act, dact, seed))
        elif type == "Convolutional":
            self.layers.append(Layers.ConvolutionalLayer(prev_nodes, *params, act, dact, seed)) # prev_nodes == input_shape
        elif type == "Pooling":
            self.layers.append(Layers.PoolingLayer(prev_nodes, *params, seed))
        else:
            raise ValueError("No valid layer type identifier input")

    def FeedForward(self, x):
        """
        Do forward pass through network
        Calls upon layers in succession, calculation handled by individual layers
        Output from each layer is then the input to the next, with the final result being saved
        a and z differ in indexing/length because a also includes initial input x, which pushes the index forwards
        """
        layers = self.layers
        L = len(layers) - 1 # zero indexed
        
        layers[0](x)
        for l in range(1, L+1, 1):
            layers[l](layers[l-1].a)

    def BackPropogate(self, t):
        """
        Use last forward pass to update weights & biases through cost function
        Calculates initial gradient using cost function, and passes derivative with respect to output a
        backwards through the layers, with each layer updating it to the next
        Layer index is also passed along, as it is used in gradient accumulation methods "RMSProp" and "Adam"
        """
        layers = self.layers
        eta = self.eta
        L = len(layers) - 1 # zero indexed

        dCda = self.costgrad(layers[L].a, t) # Output cost
        for l in range(L, -1, -1):
            #
            # HERE: Do 2D to 1D check?
            # layers[l].input_nodes = int or tuple ?
            # or output shape? Reshape 1D dCda into 2D as
            # backprop enters pooling/conv.
            #
            dCda = layers[l].update_wb(dCda, eta, l) # Each hidden layer updates with cost, and gives new to prev layer

        self.iters += 1 # Counting iterations, since some adaptive learning methods use it

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

    def test(self, X):
        """
        Calls upon network for all points in X
        """
        a = np.zeros((len(X), *self.layers[-1].output_shape), dtype=float)
        for i, Xi in enumerate(X):
            a[i] = self.__call__(Xi)

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
        a = np.zeros(len(X), dtype=float)
        for i, Xi in enumerate(X):
            a[i] = 1 if self.__call__(Xi) > 0.5 else 0

        return a

    def reset_parameters(self):
        """
        Re-initializes parameters, used when comparing outcomes for different hyperparameters
        Should return to exact same state as before, due to seeded randomness
        """
        for layer in self.layers:
            layer.reset_wb()
    
    def save_params(self, path="NetworkParameters"):
        """
        Saves weights and biases for all layers to file
        """
        wb = np.array([[l.W, l.b] for l in self.layers])
        np.save(path, wb)
    
    def load_params(self, path="NetworkParameters.npy"):
        """
        Loads weights and biases for all layers from file
        """
        wb = np.load(path, allow_pickle=True)
        for i, l in enumerate(self.layers):
            l.W, l.b = wb[i]