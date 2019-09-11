import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt
import math
import time
''' 
You are only required to fill the following functions
mean_squared_loss
mean_squared_gradient
mean_absolute_loss
mean_absolute_gradient
mean_log_cosh_loss
mean_log_cosh_gradient
root_mean_squared_loss
root_mean_squared_gradient
preprocess_dataset
main

Don't modify any other functions or commandline arguments because autograder will be used
Don't modify function declaration (arguments)

'''

def hypothesis(X , w ):
	prediction = np.dot(X,w)
	return prediction

def mean_squared_loss(xdata, ydata, weights):
	'''
	weights = weight vector [D X 1]
	xdata = input feature matrix [N X D]
	ydata = output values [N X 1]
	Return the mean squared loss
	'''
	m = ydata.shape[0]
	predicted = hypothesis(xdata,weights)
	J = (1/(2*m))*np.sum(np.square(predicted - ydata))
	return J

def mean_squared_gradient(xdata, ydata, weights):
	'''
	weights = weight vector [D X 1]
	xdata = input feature matrix [N X D]
	ydata = output values [N X 1]
	Return the mean squared gradient
	'''
	m = ydata.shape[0]
	predicted = hypothesis(xdata,weights)
	dw = (1/m)*np.dot((predicted- ydata).T,xdata)
	#db = (1/m)*np.sum((predicted- ydata))
	return dw 

def mean_absolute_loss(xdata, ydata, weights):
	m = ydata.shape[0]
	predicted = hypothesis(xdata,weights)
	mal = (1/m)*np.sum(abs(ydata-predicted))
	return mal
def mean_absolute_gradient(xdata, ydata, weights):
	predicted = hypothesis(xdata,weights)
	n = np.abs(predicted-ydata)
	d = predicted-ydata
	sign_vector = n/d
	return np.dot(sign_vector,xdata)

def mean_log_cosh_loss(xdata, ydata, weights):
	predicted = hypothesis(xdata,ydata)
	return np.log(np.cosh(np.absolute(predicted-ydata)))

def mean_log_cosh_gradient(xdata, ydata, weights):
	predicted = hypothesis(xdata,weights)
	return np.dot(np.tanh(predicted-ydata),xdata)

def root_mean_squared_loss(xdata, ydata, weights):
	return np.sqrt(mean_squared_loss(xdata,ydata,weights))

def root_mean_squared_gradient(xdata, ydata, weights):

	d = root_mean_squared_loss(xdata,ydata,weights)
	n = mean_squared_gradient(xdata,ydata,weights)
	return n/d

class LinearRegressor:

	def __init__(self,dims):
		
		# dims is the number of the features
		# You can use __init__ to initialise your weight and biases
		# Create all class related variables here
		#self.w = np.zeros(dims,dtype=float)
		limit = 1 / math.sqrt(dims)
		self.w = np.random.uniform(-1/dims,1/dims,(dims,))
	def train(self, xtrain, ytrain, loss_function, gradient_function, epoch=10, lr=.001):
		'''
		xtrain = input feature matrix [N X D]
		ytrain = output values [N X 1]
		learn weight vector [D X 1]
		epoch = scalar parameter epoch
		lr = scalar parameter learning rate
		loss_function = loss function name for linear regression training
		gradient_function = gradient name of loss function
		'''
		# You need to write the training loop to update weights here
		for i in range(0 ,epoch):
			#predicted = hypothesis(xtrain,self.w)
			cost_list = []
			ni = []
			cost = mean_squared_loss(xtrain,ytrain ,self.w)
			dw = mean_squared_gradient(xtrain,ytrain,self.w)
			self.w = self.w - lr * dw
			np.empty((2,1))
			cost_list.append(cost)
			ni.append(i)
			if (i%100 == 0):
				print("cost ===>>"+str(cost)+" :" + str(i))
		

	def predict(self, xtest):
		predicted = np.dot(xtest , self.w)
		#int_pred = predicted.astype(dtype=int)
		#print(int_pred)
		#print(predicted)
		n = xtest.shape[0]
		a =np.full(n,0,dtype=int)
		for i in range(0,n):
			a[i] = i
		#print(a.shape)
		predicted = np.where(predicted < 0 , np.random.randint(0,5) ,predicted)

		#print(predicted.shape)
		prediction = np.column_stack((a,predicted))
		#print(prediction.shape)
		np.savetxt("prediction.csv", prediction , fmt="%i , %i", delimiter="," , header="instance (id),count",comments="")

def read_dataset(trainfile, testfile):
	'''
	Reads the input data from train and test files and 
	Returns the matrices Xtrain : [N X D] and Ytrain : [N X 1] and Xtest : [M X D] 
	where D is number of features and N is the number of train rows and M is the number of test rows
	'''
	xtrain = []
	ytrain = []
	xtest = []
	trainfile = "train.csv"
	testfile = "test.csv"
	with open(trainfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtrain.append(row[:-1])
			ytrain.append(row[-1])

	with open(testfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtest.append(row)

	return np.array(xtrain), np.array(ytrain), np.array(xtest)

def get_one_hot(column, cat):
    res = np.eye(cat)[np.array(column).reshape(-1)]
    return res.reshape(list(column.shape)+[cat])

def preprocess_dataset(xdata, ydata=None):
	'''
	xdata = input feature matrix [N X D] 
	ydata = output values [N X 1]
	Convert data xdata, ydata obtained from read_dataset() to a usable format by loss function

	The ydata argument is optional so this function must work for the both the calls
	xtrain_processed, ytrain_processed = preprocess_dataset(xtrain,ytrain)
	xtest_processed = preprocess_dataset(xtest)	
	
	NOTE: You can ignore/drop few columns. You can feature scale the input data before processing further.
	'''
	all_ones = np.ones(xdata.shape[0])
	days = xdata[: , 5]
	b = np.where(days == "Sunday", 0 ,days)
	c = np.where(b == "Monday", 1 ,b)
	d = np.where(c == "Tuesday", 2 ,c)
	e = np.where(d == "Wednesday", 3 ,d)
	f = np.where(e == "Thursday", 4 ,e)
	g = np.where(f == "Friday", 5 ,f)
	days_1_to_7 = np.where(g == "Saturday", 6 ,g)
	days_1_to_7 = days_1_to_7.astype(dtype=int)

	season = xdata[: , 2].astype(dtype=int)
	hr = xdata[: , 3].astype(dtype=int)
	days_array = get_one_hot(days_1_to_7 , 7)
	season_array = norm(season)
	season_array = get_one_hot(season_array,4)
	hr_array = get_one_hot(hr , 24)


	processed_data = np.delete(xdata,[0,1,2,3,5] ,axis=1)
	processed_data = np.column_stack((all_ones,processed_data,days_array ,hr_array,season_array))
	processed_data = processed_data.astype(dtype=float)
	processed_data = processed_data.astype(dtype=float)
	print(processed_data.shape)


	if(ydata is not None):
	    new_Y = ydata.astype(float)
	    return processed_data,new_Y
	else:
		return processed_data

def norm(array):
	a = np.where(array == 1, 0 ,array)
	b = np.where(a == 2, 1 ,a)
	c = np.where(b == 3, 2 ,b)
	d = np.where(c == 4, 3 ,c)
	return d


dictionary_of_losses = {
	'mse':(mean_squared_loss, mean_squared_gradient),
	'mae':(mean_absolute_loss, mean_absolute_gradient),
	'rmse':(root_mean_squared_loss, root_mean_squared_gradient),
	'logcosh':(mean_log_cosh_loss, mean_log_cosh_gradient),
}

def main():

	# You are free to modify the main function as per your requirements.
	# Uncomment the below lines and pass the appropriate value
	start_time = time.time()
	xtrain, ytrain, xtest = read_dataset(args.train_file, args.test_file)
	xtrainprocessed, ytrainprocessed = preprocess_dataset(xtrain, ytrain)
	xtestprocessed = preprocess_dataset(xtest)
	model = LinearRegressor(xtrainprocessed.shape[1])
	# The loss function is provided by command line argument	
	loss_fn, loss_grad = dictionary_of_losses[args.loss]

	model.train(xtrainprocessed, ytrainprocessed, loss_fn, loss_grad, args.epoch, args.lr)

	ytest = model.predict(xtestprocessed)
	print("Time taken : %s seconds " % (time.time() - start_time))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--loss', default='mse', choices=['mse','mae','rmse','logcosh'], help='loss function')
	parser.add_argument('--lr', default=1.0, type=float, help='learning rate')
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--train_file', type=str, help='location of the training file')
	parser.add_argument('--test_file', type=str, help='location of the test file')

	args = parser.parse_args()

	main()
