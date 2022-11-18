import numpy as np
import matplotlib.pyplot as plt

from src.neural_network import *
from src.NN_func import *


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

np.random.seed(0)

def accuracy(y,t):
	return np.mean(np.equal(y,t))

train_size = 0.75
test_size = 1 - train_size

x = data['data']
t = np.reshape(data['target'], (-1,1))

x_train, x_test, t_train, t_test = train_test_split(x, t, train_size=train_size, test_size=test_size)

batch_size = 5
eta = 0.01
gamma = 0.001
hidden_layers = [120]*6
print(hidden_layers)
epochs = 70  #70

nn = neural_network(x_train, t_train, hidden_layers, CE(), tanh(), tanh(), eta, gamma, batch_size, epochs, "Xavier")
nn.train()
ttilde =  nn.feed_forward_out(x_test)
print("Test accuracy:",accuracy(np.where(ttilde > 0.5, 1, 0), t_test))
ttilde =  nn.feed_forward_out(x_train)
print("Train accuracy:", accuracy(np.where(ttilde > 0.5, 1, 0), t_train))


def test():
	etas = np.logspace(-1,-3,3)
	gammas = np.logspace(-1, -4, 4)
	n_layers = [4,5,6]
	n_nodes = [100,110,120,130]
	
	grid = np.zeros((len(etas), len(gammas), len(n_layers), len(n_nodes)))
	for i, eta in enumerate(etas):
		for j, gamma in enumerate(gammas):	
			for k, n_layer in enumerate(n_layers):
				for l, n_node in enumerate(n_nodes):
					layers = [n_node]*n_layer
					nn = neural_network(x_train, t_train, layers, MSE(), tanh(), tanh(), eta, gamma, batch_size, epochs, "Xavier")
					nn.train()
					ttilde =  nn.feed_forward_out(x_test)
					print("Test accuracy:",accuracy(np.where(ttilde > 0.5, 1, 0), t_test))
					grid[i,j,k,l] = accuracy(np.where(ttilde > 0.5, 1, 0), t_test)
					ttilde =  nn.feed_forward_out(x_train)
					print("Train accuracy:", accuracy(np.where(ttilde > 0.5, 1, 0), t_train))	


	xi, yi, zi, wi = np.unravel_index(np.argmax(grid), grid.shape)
	print("============================")
	print("Best values:")
	print("eta 	: %.4f" % etas[xi])
	print("gamma 	: %.4f" % gammas[yi])
	print("n_layers : %i" % n_layers[zi])
	print("n_nodes  : %i" % n_nodes[wi])
	print("accuracy : %.4f" % grid[xi,yi,zi,wi])

#test()

""" (cross-entropy)
Best values:
eta 	: 0.0100
gamma 	: 0.0010
n_layers : 4
n_nodes  : 120
accuracy : 0.9301
"""

""" (MSE)
Best values:
eta 	: 0.0100
gamma 	: 0.1000
n_layers : 4
n_nodes  : 110
accuracy : 0.9021
"""





n_layers = [3,4,5,6,7]
n_nodes  = [100,120,140,160,180]

def accuracy_grid_test(name, cost_f, act_f_hid, act_f_out, create_data, title):
	grid = np.zeros((len(n_layers),len( n_nodes)))

	if (create_data): # 1 : create new grid and save, 0: load grid 
		for i, n_l_i in enumerate(n_layers):
			for j, n_n_i in enumerate(n_nodes):
				layers = [n_n_i]*n_l_i
				print(layers)
				nn = neural_network(x_train, t_train, layers, cost_f, act_f_hid, act_f_out, eta, gamma, batch_size, epochs, "Xavier")
				nn.train()
				ttilde =  nn.feed_forward_out(x_test)
				ttilde = np.where(ttilde > 0.5, 1, 0)
				grid[i,j] = accuracy(ttilde, t_test)
		np.save("data/accuracy_layers_nodes_grid_"+name, grid)
	else:
		grid = np.load("data/accuracy_layers_nodes_grid_"+name+".npy")


	#extent = [n_layers[0], n_layers[-1], n_nodes[0], n_nodes[-1]]
	
	x, y = np.meshgrid(n_layers, n_nodes)
	plt.contourf(x,y, grid.T, levels=25)
	xi, yi = np.unravel_index(np.argmax(grid), grid.shape)
	plt.plot(n_layers[xi], n_nodes[yi], 'ro', label=r'$n_{layers} = $ %i, $n_{nodes} = $ %i, accuracy = %.3f' % (n_layers[xi], n_nodes[yi], grid[xi, yi]))
	plt.xlabel("Number of layers")
	plt.ylabel("Number of nodes in layer")
	plt.title("Accuracy score " + title)
	plt.legend(loc='upper center', bbox_to_anchor=[0.5, 1.25])
	plt.colorbar()
	plt.savefig("figures/nn_grid_test"+name+".png", bbox_inches="tight")
	plt.show()

c_data = 0

#accuracy_grid_test("ce_tanh", cost_f=CE(), act_f_hid=tanh(), act_f_out=sigmoid(), create_data=c_data, title="CE-tanh")
#accuracy_grid_test("mse_tanh", cost_f=MSE(), act_f_hid=tanh(), act_f_out=sigmoid(), create_data=c_data, title="MSE-tanh")
#accuracy_grid_test("ce_tanh_tanh", cost_f=CE(), act_f_hid=tanh(), act_f_out=tanh(), create_data=c_data, title="CE-tanh-tanh")
#accuracy_grid_test("ce_sigmoid_tanh", cost_f=CE(), act_f_hid=sigmoid(), act_f_out=tanh(), create_data=c_data, title="CE-sigmoid-tanh")






etas = np.logspace(-1, -5, 5) #[1e-1, 1e-2, 1e-3, 1e-4]
gammas = np.logspace(0, -4, 5)

# makes grid of gamma and learning rate arrays

def accuracy_grid_test_2(name, cost_f, act_f_hid, act_f_out, create_data):
	"""
	Runs the neural network over a grid of learning rates (etas=...) and 
	lambdas (gammas=...).
	
	"""
	grid = np.zeros((len(etas),len(gammas)))

	if (create_data): # 1 : create new grid and save, 0: load grid 
		for i, eta_i in enumerate(etas):
			for j, gamma_j in enumerate(gammas):
				nn = neural_network(x_train, t_train, hidden_layers, cost_f, act_f_hid, act_f_out, eta_i, gamma_j, batch_size, epochs, "Xavier")
				nn.train()
				ttilde =  nn.feed_forward_out(x_test)
				ttilde = np.where(ttilde > 0.5, 1, 0)
				grid[i,j] = accuracy(ttilde, t_test)
		np.save("data/accuracy_layers_nodes_grid_"+name, grid)
	else:
		grid = np.load("data/accuracy_layers_nodes_grid_"+name+".npy")


	#extent = [-1, -5, 0, -4]
	x, y = np.meshgrid(etas, gammas)
	plt.contourf(x, y, grid.T, levels=15)
	xi, yi = np.unravel_index(np.argmax(grid), grid.shape)
	
	plt.plot(etas[xi], gammas[yi], 'ro', label=r'$\eta =$ %.3f, $\gamma =$ %.3f, accuracy = %.3f' % (etas[xi], gammas[yi], grid[xi, yi]))
	plt.yscale("log")
	plt.xscale("log")
	plt.xlabel(r"Learning rate $\eta$")
	plt.ylabel(r"$\lambda$")
	plt.title("Accuracy score")
	plt.legend(loc='upper center', bbox_to_anchor=[0.5, 1.2])
	plt.colorbar()
	plt.savefig("figures/nn_grid_test"+name+".png", bbox_inches="tight")
	plt.show()

#accuracy_grid_test_2("ce_tanh_2", cost_f=CE(), act_f_hid=tanh(), act_f_out=sigmoid(), create_data=0)
#accuracy_grid_test_2("mse_tanh_2", cost_f=MSE(), act_f_hid=tanh(), act_f_out=sigmoid(), create_data=0)
#accuracy_grid_test_2("ce_tanh_tanh_2", cost_f=CE(), act_f_hid=tanh(), act_f_out=tanh(), create_data=1)
#accuracy_grid_test_2("mse_tanh_tanh_2", cost_f=MSE(), act_f_hid=tanh(), act_f_out=tanh(), create_data=1)



