# much help from https://github.com/mnielsen/neural-networks-and-deep-learning

import random
import numpy as np

# z = Wx + b
def sigmoid_function(z):
	return 1.0 / (1.0 + np.exp(-z))    

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
	def __init__(self, sizes):
		self.num_layers = len(sizes) # sizes is a list of sizes for each layer
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:]] # x is the size of the input layer, and y is the size of the output layer

	# x is input
	def feedforward(self, x):
		for W, b in zip(self.weights, self.biases):
			x = sigmoid(np.dot(W, x) + b)
		return x

	# training_data is a list of tuples (x, y)
	# epochs is number of epochs
	# mini_batch_size is how many training data exapmles to group into one 'mini-batch'
	# eta is learning rate for gradient descent
	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		if test_data:
			n_test = len(test_data) # if there is some test_data provided, evaluate over that after each epoch
		n = len(training_data)
		for e in range(epochs):
			np.random.shuffle(training_data)  
			mini_batches = [training_data[k:k + size] for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			print("Epoch {0} complete".format(j))

	def update_mini_batch(self, mini_batch, eta):
		total_deltas_biases = [np.zeros(b.shape) for b in self.biases]
        total_delta_weights = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			d_W, d_b = self.backprop(x, y)
			total_delta_weights = [W + dw for W, dw in zip(total_delta_weights, d_W)]
			total_delta_biases = [b + db for b, db in zip(total_delta_biases, d_b)]
		self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, total_delta_weights)]
		self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, total_delta_biases)]

	def backprop(self, x, y):
		# forward pass
        a = x
        for w, b in zip(self.weights, self.biases):
            a = np.dot(w, a) + b

        # backward pass

	
