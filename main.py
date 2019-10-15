import numpy as np
import nn
import csv
import pickle
import time

def taskXor():
	print("Training XOR...")
	XTrain, YTrain, XVal, YVal, XTest, YTest = loadXor()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	nn1 = nn.NeuralNetwork(lr=0.005, batchSize=8, epochs=50)
	# Add layers to neural network corresponding to inputs and outputs of given data
	nn1.addLayer(nn.FullyConnectedLayer(2,12,'relu'))
	#nn1.addLayer(nn.FullyConnectedLayer(12,64,'relu'))
	nn1.addLayer(nn.FullyConnectedLayer(12,2,'softmax'))	
	###############################################
	# TASK 3a (Marks 7) - YOUR CODE HERE
	
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal)
	pred, acc = nn1.validate(XTest, YTest)
	with open("predictionsXor.csv", 'w') as file:
		writer = csv.writer(file)
		writer.writerow(["id", "prediction"])
		for i, p in enumerate(pred):
			writer.writerow([i, p])
	print('Test Accuracy',acc)
	return nn1
	
def preprocessMnist(X):
	# Perform any data preprocessing that you wish to do here
	# Input: A 2-d numpy array containing an entire train, val or test split | Shape: n x 28*28
	# Output: A 2-d numpy array of the same shape as the input (If the size is changed, you will get downstream errors)
	###############################################
	# TASK 3c (Marks 0) - YOUR CODE HERE
	fac = 0.99 / 255
	processed_X = np.asfarray(X) * fac + 0.01
	return processed_X
	###############################################


def taskMnist():
	print("Training: MNIST...")
	XTrain, YTrain, XVal, YVal, XTest, _ = loadMnist()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	nn1 = nn.NeuralNetwork(lr=0.005, batchSize=8, epochs=50)
	# Add layers to neural network corresponding to inputs and outputs of given data
	nn1.addLayer(nn.FullyConnectedLayer(28*28, 256,'relu'))
	#nn1.addLayer(nn.FullyConnectedLayer(128,256,'relu'))
	nn1.addLayer(nn.FullyConnectedLayer(256,10,'softmax'))
	###############################################
	# TASK 3b (Marks 13) - YOUR CODE HERE
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal)
	pred, acc = nn1.validate(XVal, YVal)
	print(acc)
	pred, _ = nn1.validate(XTest, None)
	with open("predictionsMnist.csv", 'w') as file:
		writer = csv.writer(file)
		writer.writerow(["id", "prediction"])
		for i, p in enumerate(pred):
			writer.writerow([i, p])
	return nn1

################################# UTILITY FUNCTIONS ############################################
def oneHotEncodeY(Y, nb_classes):
	# Calculates one-hot encoding for a given list of labels
	# Input :- Y : An integer or a list of labels
	# Output :- Coreesponding one hot encoded vector or the list of one-hot encoded vectors
	return (np.eye(nb_classes)[Y]).astype(int)

def loadXor():
	# This is a toy dataset with 10k points and 2 labels.
	# The output can represented as the XOR of the input as described in the problem statement
	# There are 7k training points, 1k validation points and 2k test points
	train = pickle.load(open("data/xor/train.pkl", 'rb'))
	test = pickle.load(open("data/xor/test.pkl", 'rb'))
	testX, testY = np.array(test[0]), np.array(oneHotEncodeY(test[1],2))
	trainX, trainY = np.array(train[0][:7000]), np.array(oneHotEncodeY(train[1][:7000],2))
	valX, valY = np.array(train[0][7000:]), np.array(oneHotEncodeY(train[1][7000:],2))

	return trainX, trainY, valX, valY, testX, testY

def loadMnist():
	# MNIST dataset has 50k train, 10k val, 10k test
	# The test labels have not been provided for this task
	train = pickle.load(open("data/mnist/train.pkl", 'rb'))
	test = pickle.load(open("data/mnist/test.pkl", 'rb'))
	testX = preprocessMnist(np.array(test[0]))
	testY = None # For MNIST the test labels have not been provided
	trainX, trainY = preprocessMnist(np.array(train[0][:50000])), np.array(oneHotEncodeY(train[1][:50000],10))
	valX, valY = preprocessMnist(np.array(train[0][50000:])), np.array(oneHotEncodeY(train[1][50000:],10))

	return trainX, trainY, valX, valY, testX, testY
#################################################################################################

if __name__ == "__main__":
	start_time = time.time()
	np.random.seed(7)
	taskXor()
	taskMnist()
	print("--- %s seconds ---" % (time.time() - start_time))

