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
        """
        lim = 1e2
        x = np.where(np.less(lim, x), 1, x)
        x = np.where(np.less(x, -lim), 0, x)
        x = np.where(np.logical_and(np.less(x,lim), np.less(x, -lim)), 1/(1+ np.exp(-x)), x)
        return x
        """
        output = 1/(1 + np.exp(-x))
        output[np.isnan(output)] = 0
        return output	
	

    def deriv(self, x):
        output = np.exp(-x)/(1 + np.exp(-x))**2
        output[np.isnan(output)] = 0
        return output
	#return np.exp(-abs(x))/(1 + np.exp(-abs(x)))**2

class ELU:
    def __call__(self, x):
        alpha = 0.1
        out = np.where(np.less(x,0), alpha*np.exp(x) - 1, x)
        return out

    def deriv(self, x):
        alpha = 0.1
        out = np.where(np.less(x,0), alpha*np.exp(x), 1)
        return out

class RELU:
    def __call__(self, x):
        out = np.where(np.greater(x, 0), x, 0)
        return out
    def deriv(self, x):
        out = np.where(np.greater(x,0), 1, 0)
        return out


class LRELU: # leaky relu
    def __call__(self, x):
        out = np.where(np.greater(x,0), x, 0.01*x)
        return out
    def deriv(self, x):
        out = np.where(np.greater(x,0), 1, 0.01)
        return out

class tanh:
    def __call__(self, x):
        output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        output[np.logical_and(np.isnan(output), np.less(x,0))] = -1
        output[np.logical_and(np.isnan(output), np.greater(x,0))] = 1
        return output 

    def deriv(self, x):
        output = 4 / (np.exp(-x) + np.exp(x))**2
        output[np.isnan(output)] = 0        
        #print(output)
        return output

class I:
    def __call__(self, x):
        return x 
    
    def deriv(self, x):
        return np.ones(x.shape)
