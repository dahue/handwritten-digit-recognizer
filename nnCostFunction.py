import numpy as np
from copy import copy
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
	#NNCOSTFUNCTION Implements the neural network cost function for a two layer
	#neural network which performs classification
	#   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
	#   X, y, lmbda) computes the cost and gradient of the neural network. The
	#   parameters for the neural network are "unrolled" into the vector
	#   nn_params and need to be converted back into the weight matrices. 
	# 
	#   The returned parameter grad should be a "unrolled" vector of the
	#   partial derivatives of the neural network.
	#

	# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
	# for our 2 layer neural network
	Theta1 = nn_params[0 : (hidden_layer_size * (input_layer_size + 1))]
	Theta1.resize(hidden_layer_size, input_layer_size + 1)

	Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1) : nn_params.size]
	Theta2.resize(num_labels, hidden_layer_size + 1)

	m = np.size(X, 0)

	J = 0
	Theta1_grad = np.zeros([np.size(Theta1, 0), np.size(Theta1, 1)])
	Theta2_grad = np.zeros([np.size(Theta2, 0), np.size(Theta2, 1)])


	# Cost Function implementation
	Y = np.zeros([m, num_labels])			#5000x10
	for i in range(0,np.size(y)):
		Y[i, y[i]-1] = 1

	a_1 = np.concatenate((np.ones([m, 1]), X), axis=1)				#5000x401
	z_2 = a_1.dot(Theta1.T)				#5000x25

	a_2 = np.concatenate((np.ones([m, 1]), sigmoid(z_2)), axis=1)	#5000x26
	z_3 = a_2.dot(Theta2.T)				#5000x10
	a_3 = sigmoid(z_3)					#5000x10
	hX = a_3
	J = (1.0 / m) * np.sum(np.sum(-Y * np.log(hX) - (1 - Y) * np.log(1 - hX), 0))
	
	# Cost regularized
	Theta1_reg = copy(Theta1)
	Theta1_reg[:, 0] = 0
	Theta2_reg = copy(Theta2)
	Theta2_reg[:, 0] = 0
	J = J + (lmbda / (2*m)) * (np.sum(np.sum(Theta1_reg ** 2, 1)) + np.sum(np.sum(Theta2_reg ** 2, 1)))

	# Back propagation algorithm
	delta_3 = a_3 - Y 				#5000x10
	delta_2 = delta_3.dot(Theta2) * np.concatenate((np.ones([m, 1]), (sigmoidGradient(z_2))), axis=1)	#5000x26
	delta_2 = np.delete(delta_2, 0, 1)	#5000x25
	    
	DELTA_1 = delta_2.T.dot(a_1)		#25x401
	DELTA_2 = delta_3.T.dot(a_2)		#10x26

	Theta1_grad = DELTA_1 / m
	Theta2_grad = DELTA_2 / m

	# Compute regularization term
	reg_1 = np.concatenate((np.zeros([np.size(Theta1, 0), 1]), (lmbda / m) * np.delete(Theta1, 0, 1)), axis=1)
	reg_2 = np.concatenate((np.zeros([np.size(Theta2, 0), 1]), (lmbda / m) * np.delete(Theta2, 0, 1)), axis=1)
	# disp(size(reg_1))
	# disp(size(reg_2))

	Theta1_grad = Theta1_grad + reg_1
	Theta2_grad = Theta2_grad + reg_2

	grad = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()), axis=0)

	return J, grad



def nnCostComputation(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
	# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
	# for our 2 layer neural network
	Theta1 = nn_params[0 : (hidden_layer_size * (input_layer_size + 1))]
	Theta1.resize(hidden_layer_size, input_layer_size + 1)

	Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1) : nn_params.size]
	Theta2.resize(num_labels, hidden_layer_size + 1)

	m = np.size(X, 0)

	J = 0
	Theta1_grad = np.zeros([np.size(Theta1, 0), np.size(Theta1, 1)])
	Theta2_grad = np.zeros([np.size(Theta2, 0), np.size(Theta2, 1)])


	# Cost Function implementation
	Y = np.zeros([m, num_labels])			#5000x10
	for i in range(len(y)):
		Y[i, y[i]-1] = 1

	a_1 = np.concatenate((np.ones([m, 1]), X), axis=1)				#5000x401
	z_2 = a_1.dot(Theta1.T)				#5000x25

	a_2 = np.concatenate((np.ones([m, 1]), sigmoid(z_2)), axis=1)	#5000x26
	z_3 = a_2.dot(Theta2.T)				#5000x10
	a_3 = sigmoid(z_3)					#5000x10
	hX = a_3
	J = (1.0 / m) * np.sum(np.sum(-Y * np.log(hX) - (1 - Y) * np.log(1 - hX), 0))
	
	# Cost regularized
	Theta1_reg = copy(Theta1)
	Theta1_reg[:, 0] = 0
	Theta2_reg = copy(Theta2)
	Theta2_reg[:, 0] = 0
	J = J + (lmbda / (2*m)) * (np.sum(np.sum(Theta1_reg ** 2, 1)) + np.sum(np.sum(Theta2_reg ** 2, 1)))

	return J


def nnGradFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
	# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
	# for our 2 layer neural network
	Theta1 = nn_params[0 : (hidden_layer_size * (input_layer_size + 1))]
	Theta1.resize(hidden_layer_size, input_layer_size + 1)

	Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1) : nn_params.size]
	Theta2.resize(num_labels, hidden_layer_size + 1)

	m = np.size(X, 0)

	# Cost Function implementation
	Y = np.zeros([m, num_labels])			#5000x10
	for i in range(len(y)):
		Y[i, y[i]-1] = 1

	a_1 = np.concatenate((np.ones([m, 1]), X), axis=1)				#5000x401
	z_2 = a_1.dot(Theta1.T)				#5000x25

	a_2 = np.concatenate((np.ones([m, 1]), sigmoid(z_2)), axis=1)	#5000x26
	z_3 = a_2.dot(Theta2.T)				#5000x10
	a_3 = sigmoid(z_3)					#5000x10

	# Back propagation algorithm
	delta_3 = a_3 - Y 				#5000x10
	delta_2 = delta_3.dot(Theta2) * np.concatenate((np.ones([m, 1]), (sigmoidGradient(z_2))), axis=1)	#5000x26
	delta_2 = np.delete(delta_2, 0, 1)	#5000x25
	    
	DELTA_1 = delta_2.T.dot(a_1)		#25x401
	DELTA_2 = delta_3.T.dot(a_2)		#10x26

	Theta1_grad = DELTA_1 / m
	Theta2_grad = DELTA_2 / m

	# Compute regularization term
	reg_1 = np.concatenate((np.zeros([np.size(Theta1, 0), 1]), (lmbda / m) * np.delete(Theta1, 0, 1)), axis=1)
	reg_2 = np.concatenate((np.zeros([np.size(Theta2, 0), 1]), (lmbda / m) * np.delete(Theta2, 0, 1)), axis=1)
	# disp(size(reg_1))
	# disp(size(reg_2))

	Theta1_grad = Theta1_grad + reg_1
	Theta2_grad = Theta2_grad + reg_2

	grad = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()), axis=0)

	return grad