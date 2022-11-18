import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.cost_functions import *
from src.activation_functions import *
from src.Project1b import *
from src.neural_network import *

# Some functions
def FrankeFunction(x, y):
    """Produce terrain data"""
    N = x.shape[0]
    a = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    b = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    c = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    d = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return a + b + c + d + np.random.normal(0, 0.1,(N,N))


def FIT_OLS(xy_len, order=5):
	# make x & y grid and zip them to X 
	x = np.linspace(0,1,xy_len)
	y = np.linspace(0,1,xy_len)
	x, y = np.meshgrid(x,y)
	#X = np.zeros((xy_len*xy_len, 2))
	#X = np.array(list(zip(x.flatten(), y.flatten())))

	# make data and flatten to dimension (N,1)
	z = FrankeFunction(x,y)

	x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst = train_test_data_evenly(x, y, z, 4)
	orders = range(order+1)
	X_lrn, X_tst, beta_OLS, z_lrn_model, z_tst_model = fit_OLS(orders, x_lrn, y_lrn, z_lrn, x_tst, y_tst, z_tst)
	#z_tst_model = np.sum(z_tst_model[i] for i in orders)
	return z_tst,  z_tst_model[order]         


def run_grid_test(xy_len = 30, gamma=np.logspace(-1,-3,5), eta=np.logspace(-1.5,-3,5), hidden_layers=[30,20,10], epochs=5000, 
	init="Random", force_create_data=False, cost_func = MSE(), act_func_hid = sigmoid(), act_func_out = I(), name="test"):
	"""
	run_grid_test:
		- Runs a grid test of the neural network with set data size, hidden_layers over the number of epochs,
		momentum parametes and learning rates with given initialization of weights (init). NB: here we
		have to remember to set force_create_data to True if we want to look at a grid with a different
		set of parameters for with the same initialization.

	PARAMETERS:
		- xy_len 	(int)		: size of our data to test 
		- gamma		(array [N,])	: array of momentum parameters to test
		- eta		(array [M,])	: array of learning rates to test
		- hidden_layers	(list)		: list of nodes in each hidden layer
		- epochs 	(int)		: number of iterations to run SGD over
		- init		(str)		: choise of initialization of our weights
		- force_create_data 	(bool)	: do we want to force test to create new data?
	RETURNS:
		- mse		(array [N,M,epochs])	: grid of mse
		- xi		(int)			: index of least mse along the learning rates 
		- yi		(int)			: index of least mse along momentum parameters
		- zi		(int)			: index of least mse along epochs

	"""
	# make x & y grid and zip them to X 
	x = np.linspace(0,1,xy_len)
	y = np.linspace(0,1,xy_len)
	x, y = np.meshgrid(x,y)
	X = np.zeros((xy_len*xy_len, 2))
	X = np.array(list(zip(x.flatten(), y.flatten())))

	# make data and flatten to dimension (N,1)
	z = FrankeFunction(x,y)
	Z = z.flatten()
	#Z /= Z.max() # normalize

	# Split into train and test set
	train_size = 0.75 
	test_size = 1 - train_size
	X_train, X_test, Z_train, Z_test = train_test_split(X, Z, train_size=train_size, test_size=test_size)
	Z_train = np.reshape(Z_train, (-1,1))
	Z_test = np.reshape(Z_test, (-1,1))

	batch_size = 20 # set batch size to constant
	
	# now we create/load the data
	try: # try to load data
		mse = np.load("data/mse_grid_" + name  + ".npy")
		r2  = np.load("data/r2_grid_" + name  + ".npy")
	except:	# if this fails, the file is not there so we create one
		force_create_data=True

	if force_create_data: # <- is true if we input it (want to create new data) or if the datafile is not present
		mse = np.zeros((len(eta), len(gamma), epochs))
		r2  = np.zeros((len(eta), len(gamma), epochs))
		for j in range(len(gamma)):
			for i in range(len(eta)):
				nn = neural_network(X_train, Z_train, hidden_layers, cost_func, act_func_hid, act_func_out, eta[i], gamma[j],batch_size, epochs, init)
				mse_l, r2_l = nn.train_convergence()           
				mse[i,j] = mse_l
				r2[i,j] = r2_l        
				del nn
		np.save("data/mse_grid_" + name, mse)
		np.save("data/r2_grid_" + name, r2)
	
	# (ETA, GAMMA, EPOCHS)
	return mse, r2



def run_terrain_test(xy_len = 30, gamma=0.0, eta=0.03, hidden_layers=[30,20,10], epochs=1000, 
		init="Random", cost_func = MSE(), act_func_hid = sigmoid(), act_func_out = I(), name="test"):
	
	x = np.linspace(0,1,xy_len)
	y = np.linspace(0,1,xy_len)
	x, y = np.meshgrid(x,y)
	X = np.zeros((xy_len*xy_len, 2))
	X = np.array(list(zip(x.flatten(), y.flatten())))

	z = FrankeFunction(x,y)
	Z = z.flatten()
	

	train_size = 0.75
	test_size = 1 - train_size
	X_train, X_test, Z_train, Z_test = train_test_split(X, Z, train_size=train_size, test_size=test_size)
	#X_train = np.reshape(X_train, (-1,1))
	Z_train = np.reshape(Z_train, (-1,1))
	Z_test = np.reshape(Z_test, (-1,1))


	#z_tst_model, z_tst = fit_OLS(x, y, z, test_size)
	#MSE_ = MSE()
	#print("MSE from OLS : %:4f" % MSE_(z_tst, z_tst_model))
	
	batch_size = 20
	nn = neural_network(X_train, Z_train, hidden_layers, cost_func, act_func_hid, act_func_out, eta, gamma, batch_size, epochs, init)
	#for i in range(3000):
	#       nn.backpropagation()

	#nn.backpropagation()
	#ztilde = nn.feed_forward_out()
	nn.train()
	ztilde = nn.feed_forward_out(X_test)
	return X_test, Z_test, ztilde








def compare_to_tf(xy_len=30, hidden_layers = [50,40,30], gamma=0.0, eta=0.03):
	# Get 0.000 MSE ...
	x = np.linspace(0,1,xy_len)
	y = np.linspace(0,1,xy_len)
	x, y = np.meshgrid(x,y)
	X = np.zeros((xy_len*xy_len, 2))
	X = np.array(list(zip(x.flatten(), y.flatten())))

	# make data and flatten to dimension (N,1)
	z = FrankeFunction(x,y)
	Z = z.flatten()
	#Z /= Z.max() # normalize

	# Split into train and test set
	train_size = 0.75
	test_size = 1 - train_size
	X_train, X_test, Z_train, Z_test = train_test_split(X, Z, train_size=train_size, test_size=test_size)
	Z_train = np.reshape(Z_train, (-1,1))
	Z_test = np.reshape(Z_test, (-1,1))
		

	model = Sequential() #    hidden_layers = [50,40,30]
	# assume n_layers = 3 
	model.add(Dense(hidden_layers[0], activation='sigmoid', kernel_regularizer=regularizers.l2(gamma),input_dim=2))
	model.add(Dense(hidden_layers[1], kernel_regularizer=regularizers.l2(gamma)))
	model.add(Dense(hidden_layers[2], kernel_regularizer=regularizers.l2(gamma)))
	model.add(Dense(1, activation='linear'))
	sgd=optimizers.SGD(lr=eta)
	model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['accuracy'])
		
	model.fit(X_train, Z_train, epochs=1000, batch_size=20, verbose=0)
	test_accuracy=model.evaluate(X_test,Z_test)[1]
	print("Test accuracy : %.4f" % test_accuracy)
