"""
ACTIVATION_FUNCTIONS:
	This file contains all activation functions we can use
	in the neural network. The activation functions must
	be in the form of a class with a __call__ method that 
	contains the activation function itself and a 
	deriv method that contains the derivative of the activation
	function (see sigmoid activation function for an example/template).
"""
import numpy as np

class sigmoid:
    def __call__(self, x):
        return 1/(1 + np.exp(-x))
    
    def deriv(self, x):
        return np.exp(-x)/(1 + np.exp(-x))**2
    
class I:
    def __call__(self, x):
        return x 
    
    def deriv(self, x):
        return np.ones(x.shape)
    
