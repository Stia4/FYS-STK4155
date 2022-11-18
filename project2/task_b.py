import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits import mplot3d
from tqdm import tqdm
from imageio import imread  
from sklearn.model_selection import train_test_split
from matplotlib import ticker, cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.NN_func import *
from src.activation_functions import *
from src.cost_functions import *
from src.neural_network import *
from src.regression_functions import *
#from .project1.Project1b import *

np.random.seed(0)


figsize = (13,4)
plt.rcParams["figure.figsize"] = figsize
plt.rcParams.update({'font.size':14})

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

	gamma = np.logspace(0, -3, 5)
	eta = np.logspace(-1.5, -3, 5)
	hidden_layers = [50, 40, 30]
	epochs = 1000
	init = "Random"
	name = "sigmoid"
	
	mse, r2 = run_grid_test(xy_len, gamma, eta, hidden_layers, epochs, init, create_data, name=name)
	xi, yi, zi = np.unravel_index(np.argmin(mse), mse.shape)
	order = 5
	z_tst, z_OLS = FIT_OLS(xy_len, order)
	
	print("\n=========================================")
	print("MSE for OLS with order %i 	: %.4f" %(order,  MSE()(z_tst, z_OLS)))
	r2_f = lambda y, ytilde : 1 - np.sum((y - ytilde)**2)/np.sum((y - np.average(y))**2)
	print("R2 score for OLS with order %i 	: %.4f" % (order, r2_f(z_tst, z_OLS)))
	
	print("\n=========================================")
	print("Optimal paramteres found for MSE grid search :")
	print(r"	- MSE    : %.4f " % mse[xi,yi,zi])
	print(r"	- eta    : %.4f " % eta[xi])
	print(r"	- gamma  : %.4f " % gamma[yi])
	print(r"	- epochs : %i" % range(epochs)[zi])
	print("") 
	
	xj, yj, zj = np.unravel_index(np.argmax(r2), r2.shape)
	print("\n=========================================")
	print("Optimal paramteres found for R2 grid search :")
	print(r"	- R2     : %.4f " % r2[xj,yj,zj])
	print(r"	- eta    : %.4f " % eta[xj])
	print(r"	- lambda : %.4f " % gamma[yj])
	print(r"	- epochs : %i" % range(epochs)[zj]); 
	print("")



	anchor = (1., 1.3)
	# plot mse grid
	fig = plt.figure(figsize=figsize)
	gs = mpl.gridspec.GridSpec(1, 2)
	ax1 = fig.add_subplot(gs[0])
	ax2 = fig.add_subplot(gs[1])
	
	ax1.plot(range(epochs)[zi], eta[xi], 'ro', label=r'$n_{epochs}$ = %i, $\eta$ = %.3f, $\lambda = $ %.3e, MSE = %.3f' % (range(epochs)[zi], eta[xi], gamma[yi], mse[xi,yi,zi]))
	x1, y1 = np.meshgrid(range(epochs), eta)
	cont = ax1.contourf(x1, y1,np.log10(mse[:,yi]), levels=15)
	ax1.set_title(r"log10(MSE($n_{epochs}$, $\eta$))")
	ax1.set_ylabel("Learning rate")
	ax1.set_xlabel("Epochs")
	ax1.legend(loc='upper center', bbox_to_anchor=anchor)
	fig.colorbar(cont, ax=ax1, location='right')
	ax2.plot(range(epochs)[zi], gamma[yi], 'ro')
	#, label=r'$n_{epochs}$ = %i, $\eta$ = %.3f, $\gamma = $ %.3e, MSE = %.3f' % (range(epochs)[zi], eta[xi], gamma[yi], mse[xi,yi,zi]))
	x2, y2 = np.meshgrid(range(epochs), gamma)
	cont = ax2.contourf(x2, y2, np.log10(mse[xi]), levels=15)
	ax2.set_title("log10(MSE($n_{epochs}$, $\lambda$))")	
	ax2.set_ylabel(r"Hyper parameter")
	ax2.set_xlabel("Epochs")
	ax2.set_yscale("log")
	fig.colorbar(cont, ax=ax2, location='right')
	#plt.tight_layout()
	#plt.subplots_adjust(top=0.8)
	plt.savefig("figures/MSE_b.png",bbox_inches="tight")
	plt.show()
	
	# plot r2 grid
	fig = plt.figure(figsize=figsize)
	gs = mpl.gridspec.GridSpec(1, 2)
	ax1 = fig.add_subplot(gs[0])
	ax2 = fig.add_subplot(gs[1])

	ax1.plot(range(epochs)[zj], eta[xj], 'ro', label=r'$n_{epochs}$ = %i, $\eta$ = %.3f, $\lambda = $ %.3e, R2 = %.3f' % (range(epochs)[zj], eta[xj], gamma[yj], r2[xj,yj,zj]))
	x1, y1 = np.meshgrid(range(epochs), eta)
	cont = ax1.contourf(x1, y1, np.log10(r2[:,yj]), levels=np.linspace(-1,0,15), cmap='viridis_r')
	ax1.set_title(r"log10(R2($n_{epochs}$, $\eta$))")
	ax1.set_ylabel("Learning rate")
	ax1.set_xlabel("Epochs")
	ax1.legend(loc='upper center', bbox_to_anchor=anchor)
	fig.colorbar(cont, ax=ax1, location='right')
	
	ax2.plot(range(epochs)[zj], gamma[yj], 'ro')
	#, label=r'$n_{epochs}$ = %i, $\eta$ = %.3f, $\gamma = $ %.3e, MSE = %.3f' % (range(epochs)[zi], eta[xi], gamma[yi], mse[xi,yi,zi]))
	x2, y2 = np.meshgrid(range(epochs), gamma)
	cont = ax2.contourf(x2, y2, np.log10(r2[xj]), levels=np.linspace(-1,0,15), cmap='viridis_r')
	ax2.set_title(r"log10(R2($n_{epochs}$, $\lambda$))")
	ax2.set_ylabel(r"$\lambda$")
	ax2.set_xlabel("Epochs")
	ax2.set_yscale("log")
	fig.colorbar(cont, ax=ax2, location='right')
	#plt.tight_layout()
	plt.savefig("figures/R2_b.png", bbox_inches="tight")
	plt.show()

    
	# Run NN with optimal parameters and compare data with model
	X_test, Z_test, ztilde = run_terrain_test(xy_len, gamma[yi], eta[xi], hidden_layers, range(epochs)[zi], init, name=name)
    
	fig = plt.figure(figsize=(10,4))
	ax = fig.add_subplot(1, 2, 1, projection='3d')
	ax2 = fig.add_subplot(1, 2, 2, projection='3d')

	ax.plot_trisurf(X_test[:,0], X_test[:,1], Z_test[:,0], cmap='viridis')#, label="Test data")
	ax.set_title("Test data")
    
	ax2.plot_trisurf(X_test[:,0], X_test[:,1], ztilde[:,0], cmap='viridis')#, label="NN Model")
	ax2.set_title("NN Model")
	
	ax.view_init(30, 60)
	ax2.view_init(30, 60)
	plt.savefig("figures/terrain.png")
	#plt.legend()
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
		
