import numpy as np
from sigmoid import sigmoid
def predict(Theta1, Theta2, X):
	#PREDICT Predict the label of an input given a trained neural network
	#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	#   trained weights of a neural network (Theta1, Theta2)

	# Useful values
	m = np.size(X, 0)
	num_labels = np.size(Theta2, 0)

	# You need to return the following variables correctly 
	p = np.zeros([np.size(X, 0), 1])

	h1 = sigmoid(np.concatenate((np.ones([m, 1]), X), axis=1).dot(Theta1.T))
	h2 = sigmoid(np.concatenate((np.ones([m, 1]), h1), axis=1).dot(Theta2.T))
	p = h2.argmax(axis=1) + 1

	# =========================================================================


	return p
