import numpy as np
def sigmoid(z):
	# Compute sigmoid function
	g = 1.0 / (1.0 + np.exp(-z))
	return g
