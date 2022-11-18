import numpy as np

class neural_network:
    def __init__(self, x_data, y_data, 
                 n_layers, n_hidden_neurons, cost_func, act_func_hid, act_func_out,
                 eta, batch_size):
        """
        neural_network:
            Class for creating and training a neural network with a flexible
            number of hidden layers and a free choice of activation- and
            cost functions.
    
        PARAMETERS : 
            * x_data           (array [N,]) : The x-data of our dataset to fit to
            * y_data           (array [N,]) : The y-data ...
            * n_layers         (int)        : Number of hidden layers
            * n_hidden_neurons (list)       : The number of hidden neurons in our neural network for each
                                              hidden layer. Must be of length n_hidden_layers
            * cost_func        (class)      : Our choice of cost function when fitting
            * act_func_hid     (class)      : Our choice of activation function for the hidden layers
            * act_finc_out     (class)      : Our choice of activation function for the output layer
        
        FUNCTIONS : 
            * create_biases_and_weights : Initializes the weights usinga normal distribution 
                                          and the bias using a - .
            * feed_forward              : 
            * backpropagation           : Performs the back-propagation algorithm which "repeatedly
                                          adjusts the weights of the connections in the network"
            * 
            
        """
        assert len(n_hidden_neurons) == n_layers, "Number of hidden layers must match the length of the list of hidden neurons"
        
        np.random.seed(999)
        
        self.x_data_full = x_data
        self.y_data_full = y_data
        
        self.eta      = eta
        self.n_layers = n_layers
        self.n_inputs = len(x_data)
        self.batch_size = batch_size
        self.n_hidden_neurons = n_hidden_neurons
        
        self.cost_func = cost_func
        self.act_func_hid = act_func_hid
        self.act_func_out = act_func_out
        
        self.create_biases_and_weights()
    
    def create_biases_and_weights(self):
        """
        create_biases_and_weights:
            Initializes the hidden weights and biases in the form of lists that take us from
            one layer to the next.
        """
        self.hidden_weights = []
        self.hidden_weights.append(np.random.randn(self.n_inputs, self.n_hidden_neurons[0]))
        
        self.hidden_bias    = []
        self.hidden_bias.append(np.random.randn(self.n_hidden_neurons[0]))
        
        
        for i in range(self.n_layers-1):
            self.hidden_weights.append(np.random.randn(self.n_hidden_neurons[i], self.n_hidden_neurons[i+1]))
            self.hidden_bias.append(np.random.randn(self.n_hidden_neurons[i+1]))

        self.hidden_weights = np.array(self.hidden_weights)
        self.hidden_bias    = np.array(self.hidden_bias)
        
        
        # output weights takes us from the last hidden layer to the output layer 
        self.output_weights = np.random.randn(self.n_hidden_neurons[-1], self.n_inputs)
        self.output_bias    = np.random.randn(1, self.n_inputs)
        
    def feed_forward_out(self):
        """
        feed_forward:
            Takes us on a magical journey through the hidden layers and out the 
            output layer.
        """
        z = self.x_data
        
        for l in range(self.n_layers):
            z = z @ self.hidden_weights[l] + self.hidden_bias[l]
    
        return z @ self.output_weights + self.output_bias
    
    def feed_forward(self):
        # run through hidden layers (start from input)
        self.z = [self.x_data @ self.hidden_weights[0] + self.hidden_bias[0]]
        self.a = [self.act_func_hid(self.z[0])]
        
        for l in range(1, self.n_layers):
            # print(self.hidden_weights.shape, self.hidden_bias.shape)
            self.z.append(self.hidden_weights[l].T @ self.a[l-1] + self.hidden_bias[l])
            self.a.append(self.act_func_hid(self.z[l]))
            
        # add output layer
        self.z.append(self.output_weights.T @ self.a[-1] + self.output_bias)
        self.a.append(self.act_func_out(self.z[-1]))
            
        self.z = np.array(self.z)
        self.a = np.array(self.a)
        
    def backpropagation(self):
        """
        backpropagation:
            Propagates us back ¯\_(ツ)_/¯
        """
        self.feed_forward() # runs through and sets up z_h and a_h 
        
        delta_l = self.cost_func.deriv(self.y_data, self.z[-1])*self.act_func_out.deriv(self.a[-1])
        delta_l = (delta_l @ self.output_weights.T) * self.act_func_out.deriv(self.z[-2])
        
        for l in reversed(range(1,self.n_layers-1)): # Can go up to L as the output layer is last layer
            #print(delta_l.shape, self.hidden_weights[l+1].shape, self.z[l+1].shape)
            delta_l = (delta_l @ self.hidden_weights[l+1].T) * self.act_func_hid.deriv(self.z[l+1])
            
            self.hidden_weights[l] -= self.eta*delta_l*self.a[l-1]
            self.hidden_bias[l]    -= self.eta*delta_l[0,:]
        
        #def predict(self, X):
        
    def train(self):
        """
        train:
            Train the neural network on minibatches and return out model...
        """
        x_data = 0 # shuffle x_data_full
        y_data = 0 # shuffle y_data_full
        for j in range(0, x_data.shape[0], self.batch_size):
            self.x_data = x_data[j:j + mini_batch_size]
            self.y_data = y_data[j:j + mini_batch_size]
            
            sefl.backpropagation()
            # self.backprop(cost, X_train_shuffle[j:j+mini_batch_size], 
            # z_train_shuffle[j:j+mini_batch_size],eta,penalty)
            
        return x_data, y_data
        