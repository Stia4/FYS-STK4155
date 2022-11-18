import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from tqdm import tqdm
from imageio import imread  
from sklearn.model_selection import train_test_split
from matplotlib import ticker, cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.activation_functions import *
from src.cost_functions import *
from src.neural_network import *
from src.regression_functions import *
#from .project1.Project1b import *

np.random.seed(0)
plt.rcParams["figure.figsize"] = (14,8)
plt.rcParams.update({'font.size':14})
figsize = (12,8)

def run_test_terrain():
	x_len = 30
	y_len = 30

	x = np.linspace(0,1,x_len)
	y = np.linspace(0,1,y_len)
	x, y = np.meshgrid(x,y) 
	X = np.zeros((x_len*y_len, 2))
	X = np.array(list(zip(x.flatten(), y.flatten())))
	
	z = FrankeFunction(x,y)
	Z = z.flatten()
	#Z /= Z.max()

		
	train_size = 0.75
	test_size = 1 - train_size
	X_train, X_test, Z_train, Z_test = train_test_split(X, Z, train_size=train_size, test_size=test_size)
        #X_train = np.reshape(X_train, (-1,1))
	Z_train = np.reshape(Z_train, (-1,1))
	Z_test = np.reshape(Z_test, (-1,1))
	
	batch_size = 20
	eta = 0.03 #0.01
	gamma = 1e-5 #0.01
	hidden_layers = [50,40,30] #[30,20,10]	
	#epochs = [10, 100, 500, 1000, 1500, 2000, 2500]

	epochs = 1000 #5000 
	nn = neural_network(X_train, Z_train, hidden_layers, MSE(), sigmoid(), I(), eta, gamma, batch_size, epochs, "Random")

	#for i in range(3000):
	#	nn.backpropagation()

	#nn.backpropagation()
	#ztilde = nn.feed_forward_out()
	nn.train()
	ztilde = nn.feed_forward_out(X_test)
	
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(1, 3, 1, projection='3d')	
	ax2 = fig.add_subplot(1, 3, 2, projection='3d')
	ax3 = fig.add_subplot(1, 3, 3, projection='3d')
	#print(X_train[:,0].shape, X_train[:,1].shape, Z_train.shape)
	#ax.plot_trisurf(X_train[:,0], X_train[:,1], Z_train[:,0], cmap='viridis')
	
	ax.plot_trisurf(X_test[:,0], X_test[:,1], Z_test[:,0], cmap='viridis')
	ax2.plot_trisurf(X_test[:,0], X_test[:,1], ztilde[:,0], cmap='viridis')
	ax3.plot_trisurf(X_test[:,0], X_test[:,1], Z_test[:,0] - ztilde[:,0], cmap='viridis')
	print(np.std(Z_test[:,0] - ztilde[:,0]))
	#ax.legend()
	#plt.tight_layout()
	plt.show();

def param_test(create_data):
	xy_len = 30

	gamma = np.logspace(1,-3, 5)
	eta = np.logspace(-1.5, -3, 5)
	hidden_layers = [50,40,30] #[50, 40, 30]
	epochs = 1000
	init = "Random"
	name = "sigmoid"
	
	mse, r2 = run_grid_test(xy_len, gamma, eta, hidden_layers, epochs, init, create_data)
	xi, yi, zi = np.unravel_index(np.argmin(mse), mse.shape)
	print("\n=========================================")
	print("Optimal paramteres found for MSE grid search :")
	print(r"	- MSE    : %.4f " % mse[xi,yi,zi])
	print(r"	- eta    : %.4f " % eta[xi])
	print(r"	- gamma  : %.4f " % gamma[yi])
	print(r"	- epochs : %i \n" % range(epochs)[zi]); 
	
	xi, yi, zi = np.unravel_index(np.argmax(r2), r2.shape)
	print("\n=========================================")
	print("Optimal paramteres found for MSE grid search :")
	print(r"	- MSE    : %.4f " % r2[xi,yi,zi])
	print(r"	- eta    : %.4f " % eta[xi])
	print(r"	- gamma  : %.4f " % gamma[yi])
	print(r"	- epochs : %i \n" % range(epochs)[zi]); 
    
	plt.plot(range(epochs)[zi], eta[xi], 'ro', label=r'$n_{epochs}$ = %i, $\eta$ = %.3f, $\gamma = $ %.3e, MSE = %.3f' % (range(epochs)[zi], eta[xi], gamma[yi], mse[xi,yi,zi]))
	x, y = np.meshgrid(range(epochs), eta)
	plt.contourf(x, y, mse[:,yi,:], levels=40) 
	plt.title("MSE")	
	plt.ylabel("Learning rate")
	plt.xlabel("Epochs")
	plt.legend()
	plt.colorbar()
	plt.savefig("figures/epoch_v_eta.pdf")
	plt.show()

	x, y = np.meshgrid(range(epochs), gamma)
	plt.plot(range(epochs)[zi], gamma[yi], 'ro', label=r'$n_{epochs}$ = %i, $\eta$ = %.3f, $\gamma = $ %.3e, MSE = %.3f' % (range(epochs)[zi], eta[xi], gamma[yi], mse[xi,yi,zi]))
	plt.contourf(x, y, mse[xi], levels=40)
	plt.title("MSE")	
	plt.ylabel(r"$\gamma$")
	plt.xlabel("Epochs")
	plt.legend()
	plt.yscale("log")
	plt.colorbar()
	plt.savefig("figures/epochs_v_gamma.pdf")
	plt.show()
	
    
    
	# Run NN with optimal parameters and compare data with model
	X_test, Z_test, ztilde = run_terrain_test(xy_len, gamma[yi], eta[xi], hidden_layers, range(epochs)[zi], init, name=name)
    
	figsize = (14,8)
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(1, 2, 1, projection='3d')
	ax2 = fig.add_subplot(1, 2, 2, projection='3d')

	ax.plot_trisurf(X_test[:,0], X_test[:,1], Z_test[:,0], cmap='viridis')#, label="Test data")
	ax.set_title("Test data")
    
	ax2.plot_trisurf(X_test[:,0], X_test[:,1], ztilde[:,0], cmap='viridis')#, label="NN Model")
	ax2.set_title("NN Model")
    
	plt.legend()
	plt.show()

    


#run_test_terrain()
param_test(create_data=False)
#param_test(create_data=True)











if (0): # set to 0 to avoid input
	print("Welcome to task 2 of project 2. Here are your options for running:")
	print("[1] Test our neural network on a linear dataset")
	print("[2] Test our neural network on the Franke Function")
	print("[3] Plot/Create data for MSE as a function of the learning rate, gamma and number of epochs run")
	print("[4] All of the above")
	ans = input()
	if ans == "1":
		run_linear_test()
	elif ans == "2":
		run_test_terrain()
	elif ans == "3":
		print("Do you wish to create new data? [y/n] (You can use our premade data with [n];^))")
		ans2 = input()
		if ans2 == "y":
			param_test(1)
		elif ans2 == "n":
			param_test(0)
	elif ans == "4":
		print("Do you wish to create new data? [y/n] (You can use our premade data with [n];^))")
		ans2 = input()
		run_linear_test()
		run_terrain_test()
		if ans2 == "y":
			param_test(1)
		elif ans2 == "n":
			param_test(0)
		
