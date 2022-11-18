"""
COST_FUNCTIONS:
	This file contains all cost functions we want to test in our
	nerual network. All activation functions must be in the form
	of a class with a __call__ method that returns the activation
	function itself, and a deriv method that returns the derivative
	of the activation function.
"""
import numpy as np

class OLS:
    def __call__(self, y, ytilde):
        ytilde = X@beta
        return 1/(2*len(y)*np.sum((yi - ytilde)**2))
    
    def deriv(self, beta, X, y):
        return 2/n*X.T@(X@beta - y)
    
class MSE:
    def __call__(self, y, ytilde):
        # y is the data
        # ytilde our model
        return 1/len(y)*np.sum((y - ytilde)**2)
    
    def deriv(self, y, ytilde):
        return 2/len(y)*(ytilde - y)



