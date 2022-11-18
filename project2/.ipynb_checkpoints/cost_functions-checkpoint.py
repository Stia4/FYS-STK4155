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