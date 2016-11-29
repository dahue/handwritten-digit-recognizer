import numpy as np
from nnCostFunction import *
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from scipy import optimize
from predict import predict
# Initialization
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10


## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load trainig set
print 'Loading Data ...'
X = np.loadtxt('data1X.txt')
y = np.loadtxt('data1y.txt').astype(int)

m = np.size(X, 1)

raw_input('Press <ENTER> to continue\n')


## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print 'Loading Saved Neural Network Parameters ...'

# Load the weights into variables Theta1 and Theta2
Theta1 = np.loadtxt('weightsTheta1.txt')#25x401
Theta2 = np.loadtxt('weightsTheta2.txt')#10x26

nn_params = np.concatenate((Theta1.ravel(), Theta2.ravel()), axis=0)


## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#

print'Feedforward Using Neural Network ...'

# Weight regularization parameter (we set this to 0 here).
lmbda = 0

J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)

print 'Cost at parameters (loaded from ex4weights): {}\n(this value should be about 0.287629)'.format(J)
raw_input('Program paused. Press enter to continue.\n')


## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print 'Checking Cost Function (w/ Regularization) ...'

# Weight regularization parameter (we set this to 1 here).
lmbda = 1.0

J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)

print 'Cost at parameters (loaded from ex4weights): {}\n(this value should be about 0.383770)'.format(J)
raw_input('Program paused. Press enter to continue.\n')


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print 'Evaluating sigmoid gradient...'

sgtuple = np.array([[1, -0.5, 0, 0.5, 1]])
g = sigmoidGradient(sgtuple)
print 'Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:'
print '{}'.format(g)
raw_input('Program paused. Press enter to continue.\n')


## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print 'Initializing Neural Network Parameters ...\n'

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate((initial_Theta1.ravel(), initial_Theta2.ravel()), axis=0)


## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print'Checking Backpropagation...'

#  Check gradients by running checkNNGradients
checkNNGradients(0)

raw_input('Program paused. Press enter to continue.\n')


## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print 'Checking Backpropagation (w/ Regularization)...'

#  Check gradients by running checkNNGradients
lmbda = 3.0
checkNNGradients(lmbda)
# Also output the costFunction debugging values
debug_J, trash  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)

print 'Cost at (fixed) debugging parameters (w/ lmbda = 10): {}\n(this value should be about 0.576051)'.format(debug_J)

raw_input('Program paused. Press enter to continue.\n')


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print'Training Neural Network...'

#  You should also try different values of lmbda
lmbda = 1.0

# Create "short hand" for the cost function to be minimized
costFunction = lambda p : nnCostComputation(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
gradFunction = lambda p : nnGradFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
nn_params = optimize.fmin_cg(costFunction, x0=initial_nn_params, fprime=gradFunction, maxiter = 100)

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = nn_params[0 : (hidden_layer_size * (input_layer_size + 1))]
Theta1.resize(hidden_layer_size, input_layer_size + 1)

Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1) : nn_params.size]
Theta2.resize(num_labels, hidden_layer_size + 1)

raw_input('Program paused. Press enter to continue.\n')


## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

# print 'Visualizing Neural Network...'

# displayData(Theta1(:, 2:end))

# raw_input('Program paused. Press enter to continue.\n')

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.
pred = predict(Theta1, Theta2, X)

# np.set_printoptions(threshold=np.inf)
print 'Training Set Accuracy: {}%\n'.format(np.double(np.count_nonzero(pred==y)) / np.size(pred) * 100)
