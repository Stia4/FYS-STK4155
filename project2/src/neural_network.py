import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.cost_functions import *
from src.activation_functions import *
from src.Project1b import *

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function



class neural_network:
	def __init__(self, x_data, y_data, 
		 n_hidden_neurons, cost_func, act_func_hid, act_func_out,
		 eta, gamma, batch_size, epochs, initialization):
		"""
		neural_network:
		    Class for creating and training a neural network with a flexible
		    number of hidden layers and a free choice of activation- and
		    cost functions.

		PARAMETERS : 
		    * x_data           (array [N,]) : The x-data of our dataset to fit to
		    * y_data           (array [N,]) : The y-data ...
		    * n_hidden_neurons (list)       : The number of hidden neurons in our neural network for each
						      hidden layer. Must be of length n_hidden_layers
		    * cost_func        (class)      : Our choice of cost function when fitting
		    * act_func_hid     (class)      : Our choice of activation function for the hidden layers
		    * act_finc_out     (class)      : Our choice of activation function for the output layer
		    * eta	       (float)	    : Learning rate
		    * gamma	       (float)	    : 
		    * batch_size       (int)	    : Size of each mini batch in the SGD algorithm
		    * epochs	       (int)	    : Number of iterations to run the SGD algorithm
		    * initialization   (str)	    : Chooses the initialization method for the weights and biases

		FUNCTIONS : 
		    * create_biases_and_weights : Initializes the weights usinga normal distribution 
						  and the bias using a - .
		    * feed_forward              : 
		    * backpropagation           : Performs the back-propagation algorithm which "repeatedly
						  adjusts the weights of the connections in the network"
		    * 
		    
		"""

		np.random.seed(999)

		self.x_data_full = x_data
		self.y_data_full = y_data

		self.x_data = x_data
		self.y_data = y_data

		self.eta      = eta
		self.gamma    = gamma
		self.epochs   = epochs
		self.n_layers = len(n_hidden_neurons)
		self.n_inputs = len(x_data)
		self.initialization = initialization
	
		self.batch_size = batch_size
		self.n_hidden_neurons = n_hidden_neurons

		self.cost_func    = cost_func
		self.act_func_hid = act_func_hid
		self.act_func_out = act_func_out

		self.create_biases_and_weights()

	def create_biases_and_weights(self):
		"""
		create_biases_and_weights:
		    Initializes the hidden weights and biases in the form of lists that take us from
		    one layer to the next.
		"""
		if (self.initialization == "Random"):
			self.hidden_weights = []
			self.hidden_weights.append(np.random.randn(self.x_data_full.shape[-1], self.n_hidden_neurons[0]))


			self.hidden_bias    = []
			#self.hidden_bias.append(np.random.randn(1,self.n_hidden_neurons[0]))
			self.hidden_bias.append(np.zeros((1,self.n_hidden_neurons[0])) + 0.01)	

			for i in range(self.n_layers-1):
			    self.hidden_weights.append(np.random.randn(self.n_hidden_neurons[i], self.n_hidden_neurons[i+1]))
			    #self.hidden_bias.append(np.random.randn(1,self.n_hidden_neurons[i+1]))
			    self.hidden_bias.append(np.zeros((1,self.n_hidden_neurons[i+1])) + 0.01)		    

			#self.hidden_weights = np.array(self.hidden_weights)
			#self.hidden_bias    = np.array(self.hidden_bias)

			# output weights takes us from the last hidden layer to the output layer 
			self.output_weights = np.random.randn(self.n_hidden_neurons[-1], 1)
			#self.output_bias    = np.random.randn(1,1)
			self.output_bias    = np.zeros((1,1)) + 0.01

		elif (self.initialization == "He"):
			std_input = np.sqrt(2/self.n_inputs)
			self.hidden_weights = []
			self.hidden_weights.append(np.random.normal(0, std_input, (self.x_data_full.shape[-1], self.n_hidden_neurons[0])))
	
			self.hidden_bias = []
			self.hidden_bias.append(np.zeros((1,self.n_hidden_neurons[0])) + 0.01)

			for i in range(self.n_layers-1):
			    std_hidden = np.sqrt(2/self.n_hidden_neurons[i])
			    self.hidden_weights.append(np.random.normal(0, std_hidden, (self.n_hidden_neurons[i], self.n_hidden_neurons[i+1])))
			    self.hidden_bias.append(np.zeros((1,self.n_hidden_neurons[i+1])) + 0.01)

			std_out 		= np.sqrt(2/self.n_hidden_neurons[-1])
			self.output_weights 	= np.random.normal(0, std_out, (self.n_hidden_neurons[-1],1))
			self.output_bias 	= np.zeros((1,1)) + 0.01 	
			
		elif (self.initialization == "Xavier"):
			lim_in = np.sqrt(1/self.n_inputs)
			self.hidden_weights = []
			self.hidden_weights.append(np.random.uniform(-lim_in, lim_in, (self.x_data_full.shape[-1], self.n_hidden_neurons[0])))

			self.hidden_bias = []
			self.hidden_bias.append(np.zeros((1,self.n_hidden_neurons[0])) + 0.01)
			
	
			for i in range(self.n_layers-1):
				lim_hid = np.sqrt(1/self.n_hidden_neurons[i])
				self.hidden_weights.append(np.random.uniform(-lim_hid, lim_hid, (self.n_hidden_neurons[i], self.n_hidden_neurons[i+1])))
				self.hidden_bias.append(np.zeros((1,self.n_hidden_neurons[i+1])) + 0.01)

			lim_out  		= np.sqrt(1/self.n_hidden_neurons[-1])
			self.output_weights 	= np.random.uniform(-lim_out, lim_out, (self.n_hidden_neurons[-1],1))
			self.output_bias    	= np.zeros((1,1)) + 0.01


	def feed_forward_out(self, z):
		"""
		feed_forward:
		    Takes us on a magical journey through the hidden layers and out the 
		    output layer.
		"""

		for l in range(self.n_layers):
		    z = self.act_func_hid(z @ self.hidden_weights[l] + self.hidden_bias[l])

		return self.act_func_out(z @ self.output_weights + self.output_bias)

	def feed_forward(self):
		# run through hidden layers (start from input)
		self.z = [self.x_data @ self.hidden_weights[0] + self.hidden_bias[0]]
		self.a = [self.act_func_hid(self.z[0])]

		for l in range(1, self.n_layers):
		    #self.z.append(self.hidden_weights[l].T @ self.a[l-1] + self.hidden_bias[l])
		    self.z.append(self.a[l-1] @ self.hidden_weights[l] + self.hidden_bias[l])
		    self.a.append(self.act_func_hid(self.z[l]))		    


		# add output layer
		#self.z.append(self.output_weights.T @ self.a[-1] + self.output_bias)
		self.z.append(self.a[-1] @ self.output_weights + self.output_bias)
		self.a.append(self.act_func_out(self.z[-1]))
		
		#print(self.a)    
		#self.z = np.array(self.z)
		#self.a = np.array(self.a)

	def backpropagation(self):
		"""
		backpropagation:
		    Propagates us back ¯\_(ツ)_/¯
		"""
		self.feed_forward() # runs through and sets up z_h and a_h 

		# set up first delta_l

		cost_deriv = self.cost_func.deriv(self.y_data, self.a[-1])
		delta_l = self.act_func_out.deriv(self.z[-1])*cost_deriv
		
		#delta_l = self.a[-1] @ self.cost_func.deriv(self.y_data, self.a[-1])	
		N = len(self.y_data)		

		self.output_weights = self.output_weights - self.eta*self.a[-2].T @ delta_l \
			- self.eta*self.gamma*self.output_weights/N
		self.output_bias    = self.output_bias    -  self.eta*delta_l[0,:]
		#self.output_bias    = self.output_bias - self.eta*delta_l	
		
		delta_l = (delta_l @ self.output_weights.T) * self.act_func_out.deriv(self.z[-2])		
		self.hidden_weights[-1] = self.hidden_weights[-1] - self.eta*self.a[-3].T @ delta_l \
			- self.eta*self.gamma*self.hidden_weights[-1]/N
		self.hidden_bias[-1]    = self.hidden_bias[-1]    - self.eta*delta_l[0,:]
		#self.hidden_bias[-1]    = self.hidden_bias[-1]    - self.eta*delta_l
		
		for l in reversed(range(1,self.n_layers-1)): 
			delta_l = (delta_l @ self.hidden_weights[l+1].T) * self.act_func_hid.deriv(self.z[l])
			self.hidden_weights[l] = self.hidden_weights[l] - self.eta*self.a[l-1].T @ delta_l \
				- self.eta*self.gamma*self.hidden_weights[l]/N
			self.hidden_bias[l]    = self.hidden_bias[l]    - self.eta*delta_l[0,:]
			#self.hidden_bias[l]    = self.hidden_bias[l]    - self.eta*delta_l
			
	def train(self):
		"""
		train:
		    Train the neural network on minibatches and return out model...
		"""
		for i in tqdm(range(self.epochs)):
			RNG = np.random.default_rng(seed=i)
			ind = RNG.permutation(len(self.x_data_full))
					
			x_data_shuffle = self.x_data_full[ind] # shuffle x_data_full
			y_data_shuffle = self.y_data_full[ind] # shuffle y_data_full
			
			for j in range(0, self.x_data_full.shape[0], self.batch_size):
				self.x_data = x_data_shuffle[j:j + self.batch_size]
				self.y_data = y_data_shuffle[j:j + self.batch_size]		    
				self.backpropagation()
			
		#z_out = self.feed_forward_out(x_data_shuffle)
		#return z_out
	
	def train_convergence(self):		
		"""
		Train the neural network on minibatches and return out model...
		"""
		mse_list = []
		r2_list = []
		r2 = lambda y, ytilde : 1 - np.sum((y - ytilde)**2)/np.sum((y - np.average(y))**2) 
		
		train_size = 0.75
		test_size = 1 - train_size
		#mse = self.cost_func()
		for i in tqdm(range(self.epochs)):
			RNG = np.random.default_rng(seed=i)
			ind = RNG.permutation(len(self.x_data_full))

			x_data_shuffle = self.x_data_full[ind] # shuffle x_data_full
			y_data_shuffle = self.y_data_full[ind] # shuffle y_data_full


			X_train, X_test, Y_train, Y_test = train_test_split(x_data_shuffle, y_data_shuffle, train_size=train_size, test_size=test_size)
			#X_train = np.reshape(X_train, (-1,1))
			#Z_train = np.reshape(Z_train, (-1,1))
			#Z_test = np.reshape(Z_test, (-1,1))

			for j in range(0, X_train.shape[0], self.batch_size):
				self.x_data = X_train[j:j + self.batch_size]
				self.y_data = Y_train[j:j + self.batch_size]
				self.backpropagation()

			ytilde = self.feed_forward_out(X_test)
			mse_i = self.cost_func(Y_test, ytilde)
			mse_list.append(mse_i)
			r2_list.append(r2(Y_test, ytilde))

		return mse_list, r2_list
		#plt.plot(range(self.epochs), mse_list)
		#plt.show()
		#z_out = self.feed_forward_out(x_data_shuffle)
		#return z_out







