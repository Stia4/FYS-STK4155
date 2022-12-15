import numpy as np
from scipy.signal import convolve # Handles n-dimensional cases as well, despite separate 2d version existing

class DenseLayer:
    """
    Dense layer class to be used with NeuralNetwork class
    Used as both container for network parameters (weights and biases) and activation functions,
    and as calculator for forward sweeps, where layers are called upon to calculate each step
    """
    def __init__(self, n_prev, n_nodes, activation_function, activation_derivative, init_method, seed=0):
        """
        Inputs: # of nodes in this layer, # of nodes in previous layer, and activation function
        Initializes weights and biases
        """
        self.inp_shape = n_prev
        self.output_shape = n_nodes          # Saving output shape for later stacking layers
        self.W = np.zeros((n_prev, n_nodes)) # Weights
        self.b = np.zeros(n_nodes)           # Biases
        self.act = activation_function       # Activation function
        self.dact = activation_derivative    # Derivative of activation function
        self.seed = seed
        self.init_wb(init_method)

    def __call__(self, x):
        # Forward sweep portion for this layer
        # Takes input from previous layer, transforms it with fully connected weights,
        # adds biases, then passes it through the activation function
        if np.ndim(x) > 1: # If multidimensional input, flatten it, used in CNNs
            x = x.flatten()

        z = x @ self.W + self.b
        a = self.act(z)

        self.x = x # Storing values for later backpropogation
        self.z = z
        self.a = a

        return z, a

    def update_wb(self, dCda, eta, l):
        delta = self.dact(self.z) * dCda # delta == dCdz
        dCda = np.matmul(self.W, delta) # Need to find dCda for next layer before weights are updated

        self.W -= eta(l, 0, np.outer(self.x, delta)) # eta(layer_number, weight (0) or bias (1), gradient)
        self.b -= eta(l, 1, delta)

        return dCda

    def init_wb(self, method):
        # Initialize weights and biases
        # Need to make sure parameters are non-zero to avoid gradient explosion, and
        # different initial values to make sure nodes diverge
        # Weights are most important due to their quantity compared to biases
        # Different initialization methods are implemented, which can be useful in different situations
        RNG = np.random.default_rng(seed = self.seed) # Try to get variation in layers random distributions
        if method == 'Normal':
            # Normal distribution
            self.W += RNG.normal(0, 1, size=self.W.shape)
            self.b += RNG.normal(0, 1, size=self.b.shape)
        elif method == 'Xavier':
            # Xavier distribution
            n = np.prod(self.inp_shape)
            self.W += RNG.uniform(-1/n**0.5, 1/n**0.5, size=self.W.shape)
            self.b += RNG.uniform(-1/n**0.5, 1/n**0.5, size=self.b.shape)
        elif method == 'He':
            # He distribution
            n = np.prod(self.inp_shape)
            self.W += RNG.normal(0, 2/n, size=self.W.shape)
            self.b += RNG.normal(0, 2/n, size=self.b.shape)
        else:
            raise ValueError("Initialization method not valid, see documentation")

    def reset_wb(self):
        # Resets weights and biases to initial state, useful for comparing outcome
        # with different setups of hyperparameters
        self.W *= 0
        self.b *= 0
        self.init_wb()

class ConvolutionalLayer(DenseLayer):
    """
    Convolution for 1D input is not supported (yet?), only 2D (grayscale) or 3D (RGB)
    Is set up to handle different strides and custom padding, 'should' work well, tested for stride=2, padding=1
    All convolving is done strictly between 2D matrices, as general forms for 3D/4D matrices were hard to develop,
    could be improved in the future?
    """
    def __init__(self, inp_shape, kernel_extent, n_filters, stride, padding,
                       activation_function, activation_derivative, init_method, seed=0):
        """
        Square kernel is assumed, with same depth as input (i.e. D=3 for RGB, and D=1 for grayscale images)
        """
        if len(inp_shape) == 2: # Depth of input, set to 1 for 2D input
            D = 1
        elif len(inp_shape) == 3:
            D = inp_shape[2]
        else:
            raise ValueError("Convolutional layer input has to be either 2D or 3D")
        self.inp_shape = inp_shape

        #
        # HERE:
        # Assert input parameters (filter, padding, stride) are legal
        #

        self.W = np.zeros((n_filters, kernel_extent, kernel_extent, D)) # filter i = kernel[i]
        self.b = np.zeros(n_filters)
        self.output_shape = [(inp_shape[0] - kernel_extent + 2*padding)/stride + 1,
                             (inp_shape[1] - kernel_extent + 2*padding)/stride + 1]
        if n_filters != 1:
            self.output_shape.append(n_filters) # Only set output as 3D if relevant
        self.output_shape = np.array(self.output_shape, dtype=int)

        self.act = activation_function       # Activation function
        self.dact = activation_derivative    # Derivative of activation function
        self.n_filters = n_filters
        self.D = D
        self.stride = stride
        self.padding = padding

        self.seed = seed
        self.init_wb(init_method)

    def __call__(self, x):
        """
        Input x is either 2D or 3D array, but has to match initialization size inp_shape
        """
        p = self.padding
        s = self.stride
        z = np.zeros(self.output_shape)

        if np.ndim(x) == 2: # If 2D, make 3D anyways for consistent equations
            x = x.reshape((*x.shape, 1))
        if np.ndim(z) == 2:
            z = z.reshape((*z.shape, 1))

        x = np.pad(x, [[p,p],[p,p],[0,0]])
        W = np.rot90(self.W, 2, (1,2))

        for i in range(self.n_filters): # convolve input with filter i
            for j in range(self.D):     # at depth j (i.e. R, G, or B for color image, or filter j from last layer)
                z[..., i] += convolve(x[..., j], W[i, ..., j], mode='valid')[::s,::s].squeeze() # 'valid', since we do our own padding
            z[..., i] += self.b[i]
        z = z.squeeze()
        a = self.act(z)

        self.x = x # NB! This is the padded input, lets us skip padding again in backpropogation
        self.z = z
        self.a = a
        return z, a

    def update_wb(self, dCda, eta, l):
        ## Go from dCda to dCdz
        if np.ndim(dCda) == 1: # If 1D data is recieved, reshape into expected form, used in CNNs
            dCda = dCda.reshape(self.output_shape)
        dCdz = self.dact(self.z) * dCda
        if self.n_filters == 1:
            dCdz = dCdz.reshape(*dCdz.shape, 1) # Add filter dimension even if only 1, such that we can do dCdz[..., i] for i = 0

        ## Get dCda for next layer
        W = np.rot90(self.W, 2, (1,2))
        dCda = np.zeros(self.inp_shape)
        if np.ndim(dCda) == 2:
            dCda = dCda.reshape(*dCda.shape, 1)
        
        # dCdz has to be interwoven and padded to counteract input stride and padding
        p = self.padding
        s = self.stride
        def interweave(x):
            """
            Interweaves zeros in a matrix, example for 2x2 input:
                       [[a 0 b 0]
            [[a b]  =>  [0 0 0 0]
             [c d]] =>  [c 0 d 0]
                        [0 0 0 0]]
            Cuts off last column/row of zeros if input row/column count is odd
            """
            x_ = np.zeros((x.shape[0]*s, x.shape[1]*s, x.shape[2]))
            for i in range(x.shape[2]):
                x_[::s, ::s, i] = x[..., i]
            if s > 1 and x.shape[0] % 2: # if odd
                x_ = x_[:-p]
            if s > 1 and x.shape[1] % 2: # if odd
                x_ = x_[:, :-p]
            return x_

        mode = 'valid' if p else 'full' # ???????

        dCdz = interweave(dCdz)
        dCdz_ = np.pad(dCdz, [[p,p], [p,p], [0,0]]) # Separate name, since interwoven version is used later
        for i in range(self.n_filters):
            for j in range(self.D):
                dCda[..., j] += convolve(W[i, ..., j], dCdz_[..., i], mode=mode)

        # Update weights
        x = self.x # Already 2D->3D reshaped and zero padded from the forwards pass
        dCdz = np.rot90(dCdz, 2, (0, 1))
        dCdW = np.zeros_like(self.W)
        for i in range(self.n_filters):
            for j in range(self.D):
                dCdW[i, ..., j] += convolve(x[..., j], dCdz[..., i], mode='valid')

        self.W -= eta(l, 0, dCdW)
        self.b -= eta(l, 1, np.array([np.sum(dCdz[..., i]) for i in range(self.n_filters)]))

        return dCda

class PoolingLayer(DenseLayer):
    """
    Very simple layer, which keeps same amount of layers, doing calculations separately for each
    Options for either 'max' or 'mean' pooling
    """
    def __init__(self, inp_shape, kernel_extent, pool_type, seed=0):
        if len(inp_shape) == 2: # Depth of input, set to 1 for 2D input
            D = 1
        elif len(inp_shape) == 3:
            D = inp_shape[2]
        else:
            raise ValueError("Pooling layer input has to be either 2D or 3D")

        self.inp_shape = inp_shape
        self.output_shape = np.array([*np.ceil(np.array(inp_shape[:2]) / kernel_extent), D], dtype=int) # Divide 2 first input size dimentions by kernel size, then ceil and save as int
        self.D = D
        self.F = kernel_extent
        self.seed = seed

        if pool_type == 'max': # NB! Just max(x), not max(abs(x)), add as option?
            self.pool_func = np.max
            self.pool_cont = lambda out, x: np.where(x == out, 1, 0)
        elif pool_type == 'mean':
            self.pool_func = np.mean
            self.pool_cont = lambda out, x: np.ones_like(x)/self.F**2 # each element contributes equally, 1/n-th of the mean, not adjusted for element content
        else:
            raise ValueError("Pooling type not valid, see documentation")
        self.pool_type = pool_type

        self.W = None
        self.b = None

    def __call__(self, x):
        F = self.F # Filter size
        contribution = np.zeros(self.inp_shape) # Which elements contributed to output, e.g. position of max element
        out = np.zeros(self.output_shape)
        if self.D == 1: # If 2D system, make 3D temporarily, only need to check input since output depth is equal
            x = x.reshape(*x.shape, 1)
            out = out.reshape(*out.shape, 1)
            contribution = contribution.reshape(*contribution.shape, 1)

        # Apply pooling to input
        for k in range(self.D):
            for i in range(self.output_shape[0]): # Triple for loop, find way to broadcast?
                for j in range(self.output_shape[1]):
                    out[i,j,k] = self.pool_func(x[i*F:(i+1)*F, j*F:(j+1)*F, k])
                    contribution[i*F:(i+1)*F, j*F:(j+1)*F, k] = self.pool_cont(out[i,j,k], x[i*F:(i+1)*F, j*F:(j+1)*F, k])

        # Return to 2D if was made temp 3D before
        x = x.squeeze()
        out = out.squeeze()

        self.x = x
        self.contribution = contribution
        self.z = out
        self.a = self.z
        return out, out # Generalization of return z, a (teit)

    def update_wb(self, dCda, eta, l):
        """
        No parameters to update, so only passes along dCda
        How gradients are distributed depend on pooling method used, so a contribution matrix is
        used to keep track.
        For max pooling, contribution matrix is 1 where max element came from and 0 else, and
        for mean pooling, all elements are equal at 1/kernel_extent**2
        """
        if np.ndim(dCda) == 1:
            dCda = dCda.reshape(self.output_shape)
        dCda_next = np.zeros(self.inp_shape)

        if len(self.inp_shape) == 2: # Temp 3D if 2D
            dCda_next = dCda_next.reshape(*dCda_next.shape, 1)
            dCda = dCda.reshape(*dCda.shape, 1)

        F = self.F
        for k in range(self.D):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    dCda_next[i*F:(i+1)*F, j*F:(j+1)*F, k] = dCda[i,j,k]*self.contribution[i*F:(i+1)*F, j*F:(j+1)*F, k]

        return dCda_next.squeeze()

    ## No weights or biases are used, so these are left empty
    ## Still included for consistent form with other layers
    def init_wb(self):
        pass
    def reset_wb(self):
        pass