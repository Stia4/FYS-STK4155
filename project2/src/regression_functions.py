"""
REGRESSION_FUNCTIONS:
	This file contains all functions used in the regression
	analysis. 
"""

import numpy as np

def make_X(x, n_orders):
	"""
	make_X:
		Makes the design matrix using x with polynomial degree max = n_orders
	PARAMETERS:	
		* x	   (array [N,])	: x-coordinates of our data
		* n_orders (int)	: max polynomial degree 
	RETURNS:
		* X 	   (array [N,n_orders]) : Design matrix
	"""
	X = np.zeros((len(x),n_orders)) # Design matrix
	for i in range(n_orders):
    		X[:,i]      = x**i
	return X



def OLS(x, y, n_orders):
	""" 
	OLS: 
		Makes a model of y with n_orders number of polynomials for
		1D x. 
	PARAMETERS:
		* x 	   (array [N,]) 	: x-coordinates for our data
		* y 	   (array [N,])		: y-coordinates ... 
		* n_orders (int) 		: number of polynomial degrees to include
	RETURNS:
		* ytilde_OLS (array [N,])	: OLS model given y, x and n_orders
	"""
	X = make_X(x, n_orders)

	beta_m_OLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
	ytilde_OLS = X @ beta_m_OLS
	return ytilde_OLS



def GD(x, y, N=100000, eta=1e-2, n_orders=4, init_values=[], return_beta=False):
	"""
	GD:
	 	Makes a model of (x,y) using gradient descrent (GD).
	PARAMETERS:
		* x   		(array [N,])	: input x-coordinates 
		* y   		(array [N,])	: input y-coordinates
		* N   		(int) 		: number of iterations to run GD over
		* eta 		(float)
		* init_values 	(list)		: if given uses init_values as initial values for beta 
		* return_beta   (bool)		: if true returns beta instead of ytilde
	RETURNS:
		* ytilde_GD 			(array [N,])		: model given y, x, N, eta using gradient descent 
		* (if return_beta) beta_GD	(array [n_orders,N])	: coefficients as a function of number of iterations
	"""
	X = make_X(x, n_orders)
	n   = len(x)
	dC = lambda beta, X, y : 2/n*X.T@(X@beta - y)
	
	if return_beta:
		# if return_beta we will return the betas, so 
		# we need to record the betas as we iterate
		beta_GD = np.zeros((n_orders,N))

		if len(init_values) == 0:
			beta_GD[:,0] = np.random.randn(n_orders)
		else: 
			beta_GD[:,0] = np.array(init_values)		
		
		for i in range(N-1):
			beta_GD[:,i+1] = beta_GD[:,i] - eta_GD*dC(beta_GD[:,i])
		
		return beta_GD

	else:
		if len(init_values) == 0:
			beta_GD = np.random.randn(n_orders)
		else: 
			beta_GD = np.array(init_values)

		for i in range(N):
			beta_GD -= eta*dC(beta_GD, X, y)

		ytilde_GD = X @ beta_GD
		return ytilde_GD



def SGD(x, y, M=5, N=100000, eta=1e-1, n_orders=4):
	""" 
	SGD: 
		Makes a model of (x,y) using stochastic gradient descent (SGD).
	PARAMETERS:
		* x 	   (array [N,])	: x-coordinates of our data
		* y	   (array [N,]) : y-coordinates of our data
		* M	   (int)	: Size of each minibatch
		* N	   (int)	: Number of iterations 
		* eta	   (float)	: Learning rate
		* n_orders (int)	: Number of polynomial degrees (highest order = x^(n_orders - 1))	

	"""
	X = make_X(x, n_orders)
	n_epochs = N//M     # keep number of iterations the same 
	m = int(n/M)        # number of minibatches

	beta_SGD = np.random.randn(n_orders)

	dC = lambda beta, X, y : 2/n*X.T@(X@beta - y)

	for epoch in tqdm(range(n_epochs)):
    		for i in range(m):
        		random_index = M*np.random.randint(m)
        		xi = X_rand[random_index:random_index+M]
        		yi = y_rand[random_index:random_index+M] 
        		gradients = (1.0/M)*dC(beta_SGD, xi, yi)
        		beta_SGD = beta_SGD - eta*gradients

	ytilde_SGD = X @ beta_SGD
	return ytilde_SGD



def SGD_momentum(x, y, n_orders, N=100000, eta=1e-1, gamma=1e-1, init_value=[], return_beta=False):
	"""
	SGD_momentum:
		Makes a model of (x,y) using stochastic gradient descent with momentum.
	PARAMETERS:
		* x		(array [N,])	: x-coordinates of our data
		* y		(array [N,])	: y-coordinates of our data
		* n_orders	(int)		: number of polynomial degrees to include
		* N		(int)		: number of iterations 
		* eta		(float)		: learning rate
		* gamma		(float)		: 
		* init_value	(list)		: if given -> sets the initial values of beta to init_value
		* return_beta 	(bool)		: if True returns beta instead of ytilde
	RETURNS:
		* ytilde 			(array [N,])		: model based on (x,y)
		* (if return_beta) beta_SGD	(array [n_orders, N])	: coefficient evolutuion as a function of number of iterations
	"""
	n   = len(x)
	X   = make_X(x, n_orders)

	dC_SGD = lambda beta : 2/n*X_rand.T@(X_rand@beta - y_rand)

	v = np.zeros((n_orders,N))
	beta_SGD = np.zeros((n_orders, N))

	if len(init_value) == 0:
		# if we do not send in init_value (not updated from [])
		# then the lengths is zero and we make init_value
		# using numpy randn
		init_value = np.random.randn(4)

	beta_SGD[:,0] = init_value # set up initial values 

	for i in tqdm(range(N-1)): # engine
		beta_SGD[:,i+1] = beta_SGD[:,i] - v[:,i]
		v[:,i+1] = gamma*v[:,i] + eta*dC_SGD(beta_SGD[:,i+1])
	
	if return_beta:
		# if return_beta then we return the beta
		return beta_SGD
	else:
		# if not return_beta then we return the model (ytilde)
		ytilde = X @ beta_SGD
		return ytilde



def SGD_miniB(x, y, n_epochs=2000, M=5, t0=10, t1=50, n_orders=4, eta = 1e-1, print_beta=True):
	"""
	SGD_miniB:
		Stochastic gradient descent with minibatches and a given number of epochs.
	PARAMETERS:
		* x		(array [N,])	: x-coordinates of our data
		* y		(array [N,])	: y-coordinates of our data
		* n_epochs	(int)		: number of epochs to iterate over
		* M		(int)		: size of each minibatch
		* t0		(float)		: first parameter in learning schedule
		* t1		(float)		: second parameter inlearning schedule 
		* n_orders	(int)		: number of orders to include
		* eta		(float)		: learning rate
		* print_beta 	(bool)		: if true print betas before returning
	RETURNS:
		* ytilde_SGD 	(array [N,])	: our model given (x,y) and parameters  	
	"""
	X = make_X(x, n_orders)
	n = len(x)        # number of datapoints
	m = int(n/M)      # number of minibatches

	def learning_schedule(t):
		return t0/(t+t1)

	beta_SGD = np.random.randn(n_orders)

	for epoch in tqdm(range(n_epochs)):
		for i in range(m):
			random_index = M*np.random.randint(m)
			xi = X[random_index:random_index+M]
			yi = y[random_index:random_index+M]
			gradients = (2.0/M)* xi.T @ ((xi @ beta_SGD)-yi)
			eta = learning_schedule(epoch*m+i) # <- use tuneable learning rate here
			beta_SGD = beta_SGD - eta*gradients
	
	if print_beta:
		print("SGD w/ minibatches: ", beta_SGD)

	ytilde_SGD = X @ beta_SGD # result
	return ytilde_SGD
