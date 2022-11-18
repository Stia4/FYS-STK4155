import numpy as np

class sigmoid:
    def __call__(self,x):
        return 1/(1 + np.exp(-x))
    
    def deriv(self,x):
        return np.exp(-x)/(1 + np.exp(-x))**2
    
