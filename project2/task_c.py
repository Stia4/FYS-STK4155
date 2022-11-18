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


figsize = (12,5)
plt.rcParams["figure.figsize"] = figsize
plt.rcParams.update({'font.size':11})


def run_grid(gamma=np.logspace(1,-3,5), eta=np.logspace(-1.5,-3,5), name = "test", init="Random", act_func_hid=sigmoid(), create_data=False):
	xy_len = 30
	hidden_layers = [50,40,30] #[50, 40, 30]
	epochs = 1000

	mse, r2 = run_grid_test(xy_len, gamma, eta, hidden_layers, epochs, init, create_data, act_func_hid=act_func_hid, name=name)
	xi, yi, zi = np.unravel_index(np.argmin(mse), mse.shape)
	print("\n=========================================")
	print(r"        - MSE    : %.4f " % mse[xi,yi,zi])
	print(r"        - eta    : %.4f " % eta[xi])
	print(r"        - gamma  : %.4f " % gamma[yi])
	print(r"        - epochs : %i" % range(epochs)[zi])
	print("")

	xj, yj, zj = np.unravel_index(np.argmax(r2), r2.shape)
	print("\n=========================================")
	print("Optimal paramteres found for R2 grid search :")
	print(r"        - R2     : %.4f " % r2[xj,yj,zj])
	print(r"        - eta    : %.4f " % eta[xj])
	print(r"        - gamma  : %.4f " % gamma[yj])
	print(r"        - epochs : %i" % range(epochs)[zj]);
	print("")

	# plot mse grid	
	fig = plt.figure(figsize=figsize)
	gs = mpl.gridspec.GridSpec(1, 2)
	ax1 = fig.add_subplot(gs[0])
	ax2 = fig.add_subplot(gs[1])

	ax1.plot(range(epochs)[zi], eta[xi], 'ro', label=r'$n_{epochs}$ = %i, $\eta$ = %.3f, $\gamma = $ %.3e, MSE = %.3f' % (range(epochs)[zi], eta[xi], gamma[yi], mse[xi,yi,zi]))
	x1, y1 = np.meshgrid(range(epochs), eta)
	cont = ax1.contourf(x1, y1, mse[:,yi], levels=40)
	ax1.set_title(r"MSE($n_{epochs}$, $\eta$)")
	ax1.set_ylabel("Learning rate")
	ax1.set_xlabel("Epochs")
	ax1.legend(bbox_to_anchor=(1.2, 1.2))
	fig.colorbar(cont, ax=ax1, location='right')
	ax2.plot(range(epochs)[zi], gamma[yi], 'ro')
	#, label=r'$n_{epochs}$ = %i, $\eta$ = %.3f, $\gamma = $ %.3e, MSE = %.3f' % (range(epochs)[zi], eta[xi], gamma[yi], mse[xi,yi,zi]))
	x2, y2 = np.meshgrid(range(epochs), gamma)
	cont = ax2.contourf(x2, y2, mse[xi], levels=40)
	ax2.set_title("MSE($n_{epochs}$, $\gamma$)")
	ax2.set_ylabel(r"Momentum parameter")
	ax2.set_xlabel("Epochs")
	ax2.set_yscale("log")
	fig.colorbar(cont, ax=ax2, location='right')
	plt.tight_layout()
	plt.savefig("figures/MSE_" + name + ".pdf")
	plt.show()

	# plot r2 grid
	fig = plt.figure(figsize=figsize)
	gs = mpl.gridspec.GridSpec(1, 2)
	ax1 = fig.add_subplot(gs[0])
	ax2 = fig.add_subplot(gs[1])

	ax1.plot(range(epochs)[zj], eta[xj], 'ro', label=r'$n_{epochs}$ = %i, $\eta$ = %.3f, $\gamma = $ %.3e, R2 = %.3f' % (range(epochs)[zj], eta[xj], gamma[yj], r2[xj,yj,zj]))
	x1, y1 = np.meshgrid(range(epochs), eta)
	cont = ax1.contourf(x1, y1, r2[:,yj], levels=np.linspace(0.0, 1.0, 25))
	ax1.set_title(r"R2($n_{epochs}$, $\eta$)")
	ax1.set_ylabel("Learning rate")
	ax1.set_xlabel("Epochs")
	ax1.legend(bbox_to_anchor=(1.2, 1.2))
	fig.colorbar(cont, ax=ax1, location='right')

	ax2.plot(range(epochs)[zj], gamma[yj], 'ro')
	#, label=r'$n_{epochs}$ = %i, $\eta$ = %.3f, $\gamma = $ %.3e, MSE = %.3f' % (range(epochs)[zi], eta[xi], gamma[yi], mse[xi,yi,zi]))
	x2, y2 = np.meshgrid(range(epochs), gamma)
	cont = ax2.contourf(x2, y2, r2[xj], levels=np.linspace(0.0, 1.0, 25))
	ax2.set_title("R2($n_{epochs}$, $\gamma$)")
	ax2.set_ylabel(r"$\gamma$")
	ax2.set_xlabel("Epochs")
	ax2.set_yscale("log")
	fig.colorbar(cont, ax=ax2, location='right')
	plt.tight_layout()
	plt.savefig("figures/R2_" + name + ".pdf")
	plt.show()


def param_test(create_data):	
	order = 5
	z_tst, z_OLS = FIT_OLS(30, order)
	
	print("\n=========================================")
	print("MSE for OLS with order %i 	: %.4f" %(order,  MSE()(z_tst, z_OLS)))
	r2_f = lambda y, ytilde : 1 - np.sum((y - ytilde)**2)/np.sum((y - np.average(y))**2)
	print("R2 score for OLS with order %i 	: %.4f" % (order, r2_f(z_tst, z_OLS)))	


	#plt.rcParams["figure.figsize"] = (10,5)
	fig, ax = plt.subplots(ncols=3, figsize=(16,5), sharey=True)
	gamma = [1e-4]
	eta = [1e-2]
	hidden_layers=[50,40,30]
	epochs = 1000	
	cost_func = MSE()
	x = range(epochs)

	### SIGMOID	
	ax[0].set_title("Sigmoid")

	mse_0, r2_0 = run_grid_test(gamma=gamma, eta=eta, hidden_layers=[50,40,30], epochs=1000,
        init="Random", force_create_data=True, cost_func = MSE(), act_func_hid = sigmoid(), act_func_out = I(), name="CR_random_sigmoid")
	ax[0].plot(x, mse_0[0,0], label="Random")
	i = np.argmin(mse_0[0,0])	
	print("Min MSE for random-sigmoid : %.4f at %i epochs" % (mse_0[0,0,i], i))
	
	mse_0, r2_0 = run_grid_test(gamma=gamma, eta=eta, hidden_layers=hidden_layers, epochs=epochs, 
	init="He", force_create_data=True, cost_func=cost_func, act_func_hid=sigmoid(), name="CR_HE_sigmoid")
	ax[0].plot(x, mse_0[0,0], label="He")
	i = np.argmin(mse_0[0,0])
	print("Min MSE for He-sigmoid : %.4f at %i epochs" % (mse_0[0,0,i], i))

	mse_0, r2_0 = run_grid_test(gamma=gamma, eta=eta, hidden_layers=hidden_layers, epochs=epochs,
	init="Xavier", force_create_data=True, cost_func=cost_func, act_func_hid=sigmoid(), name="CR_Xavier_sigmoid")
	ax[0].plot(x, mse_0[0,0], label="Xavier")	
	i = np.argmin(mse_0[0,0])
	print("Min MSE for Xavier-sigmoid : %.4f at %i epochs" % (mse_0[0,0,i], i))


	### RELU
	ax[1].set_title("RELU")

	mse_1, r2_1 = run_grid_test(gamma=gamma, eta=eta, hidden_layers=hidden_layers, epochs=epochs,
	init="Random", force_create_data=True, cost_func=cost_func, act_func_hid=RELU(), name="CR_random_RELU")
	ax[1].plot(x, mse_1[0,0], label="Random")
	i = np.argmin(mse_1[0,0])
	print("Min MSE for random-RELU : %.4f at %i epochs" % (mse_1[0,0,i], i))

	mse_1, r2_1 = run_grid_test(gamma=gamma, eta=eta, hidden_layers=hidden_layers, epochs=epochs,
	init="He", force_create_data=True, cost_func=cost_func, act_func_hid=RELU(), name="CR_HE_RELU")
	ax[1].plot(x, mse_1[0,0], label="He")
	i = np.argmin(mse_1[0,0])
	print("Min MSE for He-RELU : %.4f at %i epochs" % (mse_1[0,0,i], i))

	mse_1, r2_1 = run_grid_test(gamma=gamma, eta=eta, hidden_layers=hidden_layers, epochs=epochs,
	init="Xavier", force_create_data=True, cost_func=cost_func, act_func_hid=RELU(), name="CR_Xavier_RELU")
	ax[1].plot(x, mse_1[0,0], label="Xavier")
	i = np.argmin(mse_1[0,0])
	print("Min MSE for Xavier-RELU : %.4f at %i epochs" % (mse_1[0,0,i], i))

	### LRELU
	ax[2].set_title("LRELU")

	mse_2, r2_2 = run_grid_test(gamma=gamma, eta=eta, hidden_layers=hidden_layers, epochs=epochs,
	init="Random", force_create_data=True, cost_func=cost_func, act_func_hid=sigmoid(), name="CR_random_LRELU")
	ax[2].plot(x, mse_2[0,0], label="Random")
	i = np.argmin(mse_2[0,0])
	print("Min MSE for random-LRELU : %.4f at %i epochs" % (mse_2[0,0,i], i))

	mse_2, r2_2 = run_grid_test(gamma=gamma, eta=eta, hidden_layers=hidden_layers, epochs=epochs,
	init="He", force_create_data=True, cost_func=cost_func, act_func_hid=LRELU(), name="CR_HE_LRELU")
	ax[2].plot(x, mse_2[0,0], label="He")
	i = np.argmin(mse_2[0,0])
	print("Min MSE for He-LRELU : %.4f at %i epochs" % (mse_2[0,0,i], i))

	mse_2, r2_2 = run_grid_test(gamma=gamma, eta=eta, hidden_layers=hidden_layers, epochs=epochs,
	init="Xavier", force_create_data=True, cost_func=cost_func, act_func_hid=LRELU(), name="CR_Xavier_LRELU")
	ax[2].plot(x, mse_2[0,0], label="Xavier")
	i = np.argmin(mse_2[0,0])
	print("Min MSE for Xavier-LRELU : %.4f at %i epochs" % (mse_2[0,0,i], i))
	
	ax[0].set_ylabel("MSE")
	[[axi.set_ylim(0.0,0.25), axi.grid(), axi.legend(), axi.set_xlabel("Epochs")] for axi in ax]
	plt.tight_layout()
	plt.savefig("figures/task_c.pdf")
	plt.show()  


#param_test(create_data=False)
param_test(create_data=True)

