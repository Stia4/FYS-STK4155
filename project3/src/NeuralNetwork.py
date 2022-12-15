import numpy as np
import jax
from tqdm import trange
import Layers

class NeuralNetwork:
    def __init__(self, input_nodes, init_method="Normal"):
        """
        Take input parameters and set up necessary containters and values
        """
        self.input_nodes = input_nodes # Number of input nodes, can be multidimensional
        self.layers = []               # Container for all layers in network
        self.init = init_method        # Overarching initialisation method for weights/biases, can be overruled for individual layers
        self.structure = [[], []]      # List to keep track of network structure, preset with position for cost and learnrate info

    def __call__(self, x):
        """
        Calculate output for current weights/biases, useful for when network is done developing
        """
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

        self.structure[-2] = [costfunc, costgrad]

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
        self.structure[-1] = [eta0, method, delta, rho1, rho2] # Save input to structure

    def addLayer(self, type="Dense", params=100, act="Sigmoid", alpha=None, init_method=None, seed=0):
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
        act0 = act
        if callable(act):
            act = act
            dact = jax.autograd(act)
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
            dact = lambda z: np.ones_like(z)
        elif act == "Softmax": # Output layer multi-class classification
            act  = lambda z: np.exp(z)/(np.sum(np.exp(z)))
            # dact = lambda z: act(z) - act(z)**2
            # dact = lambda z: np.sum(np.diag(act(z)) - act(z)**2, axis=0)
            dact = lambda z: np.sum(np.diag(act(z)) - np.outer(act(z), act(z)), axis=0)

        else:
            raise ValueError("No valid activation function callable or identifier input")

        ### If this is the initial hidden layer, weights are based on input nodes
        ### else, they are based on the previous layers node count
        if len(self.layers) == 0:
            prev_nodes = self.input_nodes
        else:
            prev_nodes = self.layers[-1].output_shape

        ### Select parameter initialization method, if none is chosen -> use network default
        ### See DenseLayer function init_wb() for available methods, which other layers inherit
        if init_method is None:
            init = self.init
        else:
            init = init_method

        ### Append layer of given type
        if type == "Dense":
            if np.ndim(prev_nodes) != 0: # If previous layer output is multidimensional, flatten it
               prev_nodes = np.prod(prev_nodes)
            self.layers.append(Layers.DenseLayer(prev_nodes, params, act, dact, init, seed))
        elif type == "Convolutional":
            self.layers.append(Layers.ConvolutionalLayer(prev_nodes, *params, act, dact, init, seed)) # prev_nodes == input_shape
        elif type == "Pooling":
            self.layers.append(Layers.PoolingLayer(prev_nodes, *params, seed))
        else:
            raise ValueError("No valid layer type identifier input")

        self.structure.insert(-2, [type, params, act0, alpha, init_method, seed]) # Save layer setup

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
            dCda = layers[l].update_wb(dCda, eta, l) # Each hidden layer updates with cost, and gives new to prev layer

        self.iters += 1 # Counting iterations, since some adaptive learning methods use it

    def train(self, X, t, epochs, seed=0, silent=False):
        """
        Do forward and backward sweeps for learning
        Does epochs*len(t) iterations, where X,t are shuffled for each epoch
        Assumes epochs are long, so progress bar is set on per-epoch basis if not silent
        """
        RNG = np.random.default_rng(seed)
        if silent:
            range_ = range
        else:
            range_ = trange

        # Make 'one-hot vector', if t = 2, n_output = 4 -> t = [0, 0, 1, 0] (zero indexed)
        if self.layers[-1].output_shape > 1 and np.ndim(t[0]) == 0:
            t_ = np.zeros((len(t), self.layers[-1].output_shape)) # shape len(t) -> (len(t), output_nodes)
            for i, ti in enumerate(t):
                t_[i, ti] = 1
            t = t_

        for epoch in range(epochs):
            p = RNG.permutation(len(t))
            X_, t_ = X[p], t[p]
            for i in range_(len(t)):
                self.FeedForward(X_[i])
                self.BackPropogate(t_[i])

    def test(self, X, classify=0):
        """
        Calls upon network for all points in x, but also uses binary classification,
        where 0 is returned if output(xi) in X is less than (or equal to) 0.5, or 1 if output is larger than 0.5

        Would add functionality for multiple category translation here in
        future, i.e. [0.2, 0.2, 0.5, 0.1] -> 3 (2 for zero indexing), currently assumes
        only one node (binary problem)

        Merge with 'test' function, adding 'classify' as keyword? classify could be number of categories?
        Regression: 0 (default), Binary or single class: 1 (binary/class separated by # of output nodes), Multiclass: n >= 2
        """
        output_dim = self.layers[-1].output_shape
        if np.ndim(output_dim) == 1: # Test if output has to be unpacked or not for array generation (list or int)
            a = np.zeros((len(X), *output_dim), dtype=float)
        else:
            a = np.zeros((len(X), output_dim), dtype=float)
        a = a.squeeze()
        
        if not classify: # No classification: Simply return output
            categorize = lambda x: x
        elif classify == 1: # Single category: If several nodes: Return node idx with highest value, if single node: Return 1 if output >= 0.5, else 0
            if np.ndim(output_dim) == 1: # Assumed output layer is 1D or 0D (single value)
                categorize = lambda x: np.argmax(x)
            else: # Assumed ndim(output_dim) = 0 (single value)
                categorize = lambda x: 1 if x >= 0.5 else 0
        elif classify >= 2: # Multiple categories, return indexes of n=classify largest values from output
            categorize = lambda x: np.argpartition(x, -classify)[-classify:]
        else:
            raise ValueError("Classify argument has to be integer >= 0, see documentation")

        for i, Xi in enumerate(X):
            a[i] = categorize(self.__call__(Xi))

        return a

    def reset_parameters(self):
        """
        Re-initializes parameters, used when comparing outcomes for different hyperparameters
        Should return to exact same state as before, due to seeded randomness
        """
        for layer in self.layers:
            layer.reset_wb()
    
    def save_params(self, path="NetworkParameters.npy", silent=False):
        """
        Saves weights and biases for all layers to file
        """
        wb = np.array([[l.W, l.b] for l in self.layers], dtype=object)
        np.save(path, wb)
        if not silent:
            print(f"Saved parameters to file {path}")
    
    def load_params(self, path="NetworkParameters.npy", silent=False):
        """
        Loads weights and biases for all layers from file
        """
        wb = np.load(path, allow_pickle=True)
        for i, l in enumerate(self.layers):
            l.W, l.b = wb[i]
        if not silent:
            print(f"Loaded parameters from file {path}")

    def setup_network(self, structure):
        """
        Network structure is essentially a list of inputs to addLayer,
        followed by inputs to set_CostFunc and set_LearningRate
        Not intended to be used by user, as functions could be called individually,
        but rather as a way of automating loading networks
        For what all the inputs mean, see individual functions addLayer,
        set_CostFunc, and set_LearningRate.

        Example structure:
        structure = [
            [ "Convolutional", [3, 16, 1, 1],  "linear", None, None, 0],
            [       "Pooling",    [2, "max"],      None, None, None, 0],
            [ "Convolutional", [3, 32, 1, 1],  "linear", None, None, 0],
            [       "Pooling",    [2, "max"],      None, None, None, 0],
            [ "Convolutional", [3, 64, 1, 1],  "linear", None, None, 0],
            [       "Pooling",    [2, "max"],      None, None, None, 0],
            [         "Dense",           100, "Sigmoid", None, None, 0],
            [         "Dense",             1, "Sigmpod", None, None, 0],
            ["MSE", None],
            [1e-3, None, 1e-7, 0.9, 0.999]
        ]
        """
        for layer in structure[:-2]:
            self.addLayer(*layer)
        self.set_CostFunc(*structure[-2])
        self.set_LearningRate(*structure[-1])

    def save_network(self, path="Network.npy", silent=False):
        """
        Saves all info about network to file, including structure (layers, cost, and learnrate)
        and parameters (weights & biases).
        """
        wb = np.array([[l.W, l.b] for l in self.layers], dtype=object)
        np.save(path, np.array([self.structure, wb], dtype=object))
        if not silent:
            print(f"Saved network to file {path}")

    def load_network(self, path="Network.npy", silent=False):
        """
        Loads info as saved via save_network(), and sets it up as specified
        """
        structure, wb = np.load(path, allow_pickle=True)
        self.setup_network(structure)
        for i, l in enumerate(self.layers):
            l.W, l.b = wb[i]
        if not silent:
            print(f"Network loaded from file {path}")