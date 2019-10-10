import numpy as np

class NeuralNetwork:

	def __init__(self, lr, batchSize, epochs):
		# Method to initialize a Neural Network Object
		# Parameters
		# lr - learning rate
		# batchSize - Mini batch size
		# epochs - Number of epochs for training
		self.lr = lr
		self.batchSize = batchSize
		self.epochs = epochs
		self.layers = []


	def addLayer(self, layer):
		# Method to add layers to the Neural Network
		self.layers.append(layer)

	def get_batch(self,X, y, batch_size):
		for i in np.arange(0, X.shape[0], batch_size):
			yield(X[i:i+batch_size,:],y[i:i+batch_size,:])

	def train(self, trainX, trainY, validX=None, validY=None):
		# Method for training the Neural Network
		# Input
		# trainX - A list of training input data to the neural network
		# trainY - Corresponding list of training data labels
		# validX - A list of validation input data to the neural network
		# validY - Corresponding list of validation data labels
		
		# The methods trains the weights and baises using the training data(trainX, trainY)
		# Feel free to print accuracy at different points using the validate() or computerAccuracy() functions of this class
		###############################################
		# TASK 2c (Marks 0) - YOUR CODE HERE
		for i in range(self.epochs):
			error = 0
			#trainY = trainY.reshape(trainY.shape[0],1)
			training_data =  np.concatenate((trainX, trainY), axis=1)
			print("Tr_data"+str(training_data.shape) )
			n = len(training_data)
			# batches = [ training_data[k:k+self.batchSize] for k in range(0, n, self.batchSize)]
			# for mini_batch in batches:
			output = None
			for batchX,batchY in self.get_batch(trainX , trainY , self.batchSize):
				mini_batch = np.concatenate((batchX, batchY), axis=1)
				prediction = self.predict(batchX)
				print(prediction)
				#ce_loss = self.crossEntropyLoss(batchY , prediction)
				ce_delta = self.crossEntropyDelta(batchY ,prediction)
				error = ce_delta
				for layer in reversed(self.layers):
					error = layer.backwardpass(layer.X , error)
					layer.updateWeights(self.lr)
				print("backward pass completed")

		###############################################

	def crossEntropyLoss(self, Y, predictions):
		# Input 
		# Y : Ground truth labels (encoded as 1-hot vectors) | shape = batchSize x number of output labels
		# predictions : Predictions of the model | shape = batchSize x number of output labels
		# Returns the cross-entropy loss between the predictions and the ground truth labels | shape = scalar
		###############################################
		# TASK 2a (Marks 3) - YOUR CODE HERE
		#return (-1)*np.mean(np.dot(Y,np.log(predictions).T))
		loss = -np.sum(np.sum(Y*np.log(predictions),axis=1),axis=0)
		return loss
		###############################################

	def crossEntropyDelta(self, Y, predictions):
		# Input 
		# Y : Ground truth labels (encoded as 1-hot vectors) | shape = batchSize x number of output labels
		# predictions : Predictions of the model | shape = batchSize x number of output labels
		# Returns the derivative of the loss with respect to the last layer outputs, ie dL/dp_i where p_i is the ith 
		#		output of the last layer of the network | shape = batchSize x number of output labels
		###############################################
		# TASK 2b (Marks 3) - YOUR CODE HERE
		loss_delta = (-1) * np.divide(Y,predictions)
		return loss_delta
		###############################################
		
	def computeAccuracy(self, Y, predictions):
		# Returns the accuracy given the true labels Y and final output of the model
		correct = 0
		for i in range(len(Y)):
			if np.argmax(Y[i]) == np.argmax(predictions[i]):
				correct += 1
		accuracy = (float(correct) / len(Y)) * 100
		return accuracy

	def validate(self, validX, validY):
		# Input 
		# validX : Validation Input Data
		# validY : Validation Labels
		# Returns the predictions and validation accuracy evaluated over the current neural network model
		valActivations = self.predict(validX)
		pred = np.argmax(valActivations, axis=1)
		if validY is not None:
			valAcc = self.computeAccuracy(validY, valActivations)
			return pred, valAcc
		else:
			return pred, None

	def predict(self, X):
		# Input
		# X : Current Batch of Input Data as an nparray
		# Output
		# Returns the predictions made by the model (which are the activations output by the last layer)
		# Note: Activations at the first layer(input layer) is X itself		
		activations = X
		for l in self.layers:
			activations = l.forwardpass(activations)
		return activations






class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes, activation):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		self.activation = activation
		self.X = None
		# Stores a quantity that is computed in the forward pass but actually used in the backward pass. Try to identify
		# this quantity to avoid recomputing it in the backward pass and hence, speed up computation
		self.data = None

		# Create np arrays of appropriate sizes for weights and biases and initialise them as you see fit
		###############################################
		# TASK 1a (Marks 0) - YOUR CODE HERE
		self.weights = np.random.rand(in_nodes, out_nodes) - 0.5
		self.biases = np.random.rand(1, out_nodes) - 0.5
		#self.output = np.random.rand(in_nodes, out_nodes) - 0.5
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary
		
		# Store the gradients with respect to the weights and biases in these variables during the backward pass
		self.weightsGrad = None
		self.biasesGrad = None

	def relu_of_X(self, X):
		# Input
		# data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
		# Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
		# This will only be called for layers with activation relu
		###############################################
		# TASK 1b (Marks 1) - YOUR CODE HERE
		return np.maximum(0,X , dtype=float)
		###############################################

	def gradient_relu_of_X(self, X, delta):
		# Input
		# data : Output from next layer/input | shape: batchSize x self.out_nodes
		# delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
		# Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
		# This will only be called for layers with activation relu amd during backwardpass
		###############################################
		# X[X <= 0] = 0
		# X[X > 0] = 1
		# print(X)
		# print(delta)
		# return np.multiply(X , delta)
		# dx = np.zeros(X.shape)
		# dx[X>0] = 1
		# dx = delta * dx
		return delta*(X>0)
		
		###############################################

	def softmax_of_X(self, X):
		# Input
		# data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
		# Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
		# This will only be called for layers with activation softmax
		###############################################
		# TASK 1c (Marks 3) - YOUR CODE HERE
		s = np.exp(X)
		self.softmax =  s / np.sum(s)
		return self.softmax
		###############################################

	def gradient_softmax_of_X(self, X, delta):
		# Input
		# X  <-- softmax(input)
		# data : Output from next layer/input | shape: batchSize x self.out_nodes
		# delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
		# Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
		# This will only be called for layers with activation softmax amd during backwardpass
		# Hint: You might need to compute Jacobian first
		o1 = np.ones((X.shape[0], X.shape[1]))
		o2 = np.ones((X.shape[1], X.shape[0]))
		J = np.dot(X.T , o1)*(np.eye(X.shape[1])- np.dot(o2,X))
		return np.dot(delta,J)
		###############################################
		#TASK 1f (Marks 7) - YOUR CODE HERE
		# J = -np.outer(X, X) + np.diag(X.flatten())
		# return np.dot(delta,J)
	
		###############################################

	def forwardpass(self, X):
		# Input
		# activations : Activations from previous layer/input | shape: batchSize x self.in_nodes
		# Returns: Activations after one forward pass through this layer | shape: batchSize x self.out_nodes
		# You may need to write different code for different activation layers
		###############################################
		# TASK 1d (Marks 4) - YOUR CODE HERE
		self.X = X
		if self.activation == 'relu':
			self.data = np.dot(X, self.weights) + self.biases
			self.data = self.relu_of_X(self.data)

		elif self.activation == 'softmax':
			self.data = np.dot(X, self.weights) + self.biases
			self.data = self.softmax_of_X(self.data)
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()
		return self.data
		###############################################

	def backwardpass(self, activation_prev, delta):
	# Input
	# activation_prev : Output from the previous layer / input to the current layer | shape: batchSize x self.in_nodes
	# delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
	# Output
	# new_delta : del_Error/ del_activation_prev | shape: batchSize x self.in_nodes
	# You may need to write different code for different activation layers
	###############################################
		# TASK 1g (Marks 6) - YOUR CODE HERE
		if self.activation == 'relu':
			#print("f_start")
			inp_delta = self.gradient_relu_of_X(self.data, delta) #dE/dz 
			new_delta = np.dot(inp_delta , self.weights.T)   #dE/dX
			#print(new_delta)
			self.weightsGrad = np.dot(activation_prev.T,inp_delta)
			self.biasesGrad = np.sum(inp_delta, axis = 0)
			#print(self.weightsGrad)
			#print("wt updated")
			#print("bias updated")
			#print(self.biasesGrad)
			return new_delta
		elif self.activation == 'softmax':
			inp_delta = self.gradient_softmax_of_X(self.data, delta)
			#print("inp_del==>> "+str(inp_delta))
			# #print(inp_delta.shape)
			# print("a==>> " +str(activation_prev))
			# print(self.weights.shape)
			new_delta = np.dot(inp_delta , self.weights.T)
			self.weightsGrad = np.dot(activation_prev.T,inp_delta)
			self.biasesGrad = np.sum(inp_delta, axis = 0)
			# print(new_delta)
			# print("b_end")
			# out_delta = np.dot(self.weights.T,inp_delta)

			return new_delta
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()
		###############################################
		
	def updateWeights(self, lr):
		# Input
		# lr: Learning rate being used
		# Output: None
		# This function should actually update the weights using the gradients computed in the backwardpass
		###############################################
		# TASK 1h (Marks 2) - YOUR CODE HERE
		self.weights = self.weights - lr * self.weightsGrad
		self.biases = self.biases - lr * self.biasesGrad
		###############################################
		