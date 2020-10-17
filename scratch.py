import numpy as np
import math
import random
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt
np.random.seed(0)

class MyPreProcessor():
	def __init__(self):
		pass

	def pre_process(self, dataset):
		x,y = [],[]
		if dataset == 0: # Implement for the abalone dataset
			file = open("AbaloneDataset.txt","r") # Opening the file
			row = file.readline()
			while row:
				if row[0]=="M": # Making 3 columns M,F,I instead of 1
					row = row[1:] + " 1 0 0"
				elif row[0]=="F":
					row = row[1:] + " 0 1 0"
				else:
					row = row[1:] + " 0 0 1"

				x.append(list(map(float,row.split()))) # X array
				y.append(x[-1].pop(-4)) # Target values
				row = file.readline()
			file.close()

		elif dataset == 1: # Implement for the video game dataset
			df = pd.read_csv("GameDataset.csv")
			df = df[["Critic_Score","User_Score","Global_Sales"]] # Important columns
			df = df.dropna() # Removing NaN values
			df = df[df.User_Score!="tbd"] # Removing tbd values
			df = df.sample(frac=1,random_state = 0) # Shuffling the rows
			for i in df:
				count = 0
				for j in df[i]:
					if i=="Global_Sales":
						y.append(float(j)) # Target values
					elif i=="Critic_Score":
						x.append([])
						x[-1].append(float(j))
					else:
						x[count].append(float(j))
						count += 1

		elif dataset == 2: # Implement for the banknote authentication dataset
			file = open("banknote.txt","r")
			row = file.readline()
			while row:
				x.append(list(map(float,row.split(",")))) # X array
				row = file.readline()
			file.close()
			random.shuffle(x) # Shuffling the data
			for i in range(len(x)):
				y.append(x[i].pop()) # Creating target array

		return x, y
class MyLinearRegression():

	def __init__(self):
		self.theta = [] # Store the final values of theta after training
		self.e = [] # List to store the error values for each iteration
		self.iterations = 1000 # Set the number of epochs
		self.learning_rate = 0.01 # Learning rate alpha
		self.dtrain = {} # Dictionary to store error values on training data for each iteration
		self.dtest = {} # Dictionary to store error values on testing data for each iteration

	def hypothesis(self,x,theta): # value of y for given x and theta. Parameter reqd - x and theta
		y = np.matmul(x,theta) # Calculating predicted value
		return y

	def errorRMSE(self,x,y,theta): # Calculating RMSE. Parameters required - features, output array, theta
		diff = np.subtract(self.hypothesis(x,theta),y) # pred-act value
		e = np.sum(np.square(diff)) # sum of squares of error
		return ((e/len(y))**0.5) # RMSE error

	def errorMAE(self,x,y,theta): # Calculating MAE. Parameters reqd - input array, output array features
		diff = np.absolute(np.subtract(self.hypothesis(x,theta),y)) # absolute of pred-act value
		return (np.sum(diff)/len(y)) # MAE error

	def calcRMSE(self,x,y,theta): # Calculate gradient with current theta values
		diff = np.subtract(self.hypothesis(x, theta),y) # Predicted-actual values
		value = (np.matmul(np.transpose(x),diff))/(np.dot(np.transpose(diff),diff)**0.5) # Final values of the gradient
		return value

	def calcMAE(self,x,y,theta): # Calculate gradient with current theta
		diff = np.sign(np.subtract(self.hypothesis(x, theta),y)) # Predicted-actual value
		value = np.matmul(np.transpose(x),diff) # Final value of gradient for MAE
		return value

	def fitRMSE(self,x,y): # Training fit function for RMSE which calculates final values of theta
		e = [] # List of errors in each iteration
		theta = np.zeros((len(x[0]),1)) # List of weights initialsed with all zeroes.
		for epochs in range(self.iterations): 
			value = self.calcRMSE(x,y,theta) # Gradient calculation
			e.append(self.errorRMSE(x,y,theta)) # Error with given weights
			self.dtrain[epochs+1] = e[-1] # Error for each iteration. USED IN PLOTTING
			# self.dtest[epochs+1] = self.errorRMSE(xtest,ytest,theta) # Remove it as comment for training and validation plots
			value *= (self.learning_rate/(len(y)**0.5))
			theta = np.subtract(theta,value) # Updating the value of theta
		e.append(self.errorRMSE(x,y,theta)) # Final error
		return theta,e # Returning final parameters and error list.

	def fitMAE(self,x,y): # Fit function for MAE model
		e = [] # List of errors in each iteration
		theta = np.zeros((len(x[0]),1)) # List of weights initialised with all zeroes
		for epochs in range(self.iterations):
			value = self.calcMAE(x,y,theta) # Gradient Calculation
			e.append(self.errorMAE(x,y,theta)) # Error with given weights
			self.dtrain[epochs+1] = e[-1] # For plotting the graph
			# self.dtest[epochs+1] = self.errorMAE(xtest,ytest,theta) # Remove it as comments for training and validation loss
			value *= (self.learning_rate/(len(y)))
			theta = np.subtract(theta,value) # Parameters updated
		e.append(self.errorMAE(x,y,theta))

		return theta,e

	def fit(self, x, y, loss="rmse"): # Fit function which decides the LOSS function to be used. Parameters - Numpy input array, Numpy output array, string to specify the loss function to be used.
		x, y = x[:], y[:] # Copying the array
		x = np.concatenate((x,np.ones((len(x),1))),axis=1) # Adding a column of 1s to X array. Example - "To take care of c in case of y = mx + c"
		y = np.reshape(y,(-1,1)) # Conversion of 1D array to 2D

		if (loss.lower()=="rmse"): # Default Loss Function
			self.theta, self.e = self.fitRMSE(x,y)
		else: # In case user wants to use MAE
			self.theta, self.e = self.fitMAE(x,y)

		# print (self.theta)
		return self

	def predict(self, X): # Predicting the output array for a given input array
		X = X[:] # Copying the array
		X = np.concatenate((X,np.ones((len(X),1))),axis=1) # Adding a column of 1s
		return self.hypothesis(X,self.theta) # Returning final values of y

	def plot(self): # Ploting the error vs iteration graph for both training and validation sets.
		x,y1,y2 = [],[],[]
		for i in range(1,len(self.dtrain)+1):
			x.append(i)
			y1.append(self.dtrain[i])
			y2.append(self.dtest[i])
		plt.plot(x,y1,label="training") # Labels the line as training
		plt.plot(x,y2,label="validation") # Labels the line as testing
		plt.legend()
		plt.show()

	def normal_equation(self, x, y): # To get the parameters directly
		return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.transpose(x)),y) # Formula for normal equtation
		# return np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.matmul(np.transpose(x),y))
class MyLogisticRegression():

	def __init__(self):
		self.theta = [] # Store the final values of theta after training
		self.e = [] # List to store the error values for each iteration
		self.iterations = 1000 # Set the number of epochs
		self.learning_rate = 0.01 # Learning rate alpha
		self.dtrain = {} # Dictionary to store error values on training data for each iteration
		self.dtest = {} # Dictionary to store error values on testing data for each iteration


	def hypothesis(self, x, theta): # value of y for given x and theta. Parameter reqd - x and theta
		return 1/(1+math.e**(-1*np.matmul(x,theta))) # Calculating predicted value

	def error(self, x, y, theta): # Calculating error for given x, y and theta.
		e = np.matmul(np.transpose(np.log(0.000001+self.hypothesis(x,theta))),y) + np.matmul(np.transpose(np.log(1-self.hypothesis(x,theta)+0.000001)),1-y) # Loss function. 10^-6 added to remove the possiblity of 0 in log.
		return (-np.sum(e)/len(y)) # Return the error

	def calcBGD(self, x, y, theta): # Calculating gradient value for BGD
		diff = np.subtract(self.hypothesis(x, theta),y) # Predicted-actual values
		value = (np.matmul(np.transpose(x),diff)) # Final values of the gradient
		return value

	def calcSGD(self, x, y, theta): # Calculatin gradient value for SGD
		rand = random.randint(0,len(y)-1) # To pick a random datapoint 
		diff = np.subtract(self.hypothesis(x[rand], theta),y[rand]) # Predicted-actual value
		value = np.dot(np.reshape(np.transpose(x[rand]),(-1,1)),np.reshape(diff,(-1,1))) # Final value of gradient 
		return value

	def fitBGD(self, x, y):
		e = [] # List of errors in each iteration
		theta = np.zeros((len(x[0]),1)) # List of weights
		for epochs in range(self.iterations):
			value = self.calcBGD(x,y,theta) # Gradient Calculation
			e.append(self.error(x,y,theta)) # Error with given weights
			self.dtrain[epochs+1] = e[-1] # For plotting the graph
			# self.dtest[epochs+1] = self.error(xval, yval, theta) # Comment in for Ploting
			value *= (self.learning_rate/(len(y)))
			theta = np.subtract(theta,value) # Updating the theta
		e.append(self.error(x,y,theta))
		return theta,e

	def fitSGD(self, x, y):
		e = [] # List of errors in each iteration
		theta = np.zeros((len(x[0]),1)) # List of weights
		for epochs in range(self.iterations):
			value = self.calcSGD(x,y,theta) # Gradient Calculation
			e.append(self.error(x,y,theta)) # Error with given weights
			self.dtrain[epochs+1] = e[-1] # For plotting the graph
			# self.dtest[epochs+1] = self.error(xval, yval, theta) # Comment in for plotting
			value *= (self.learning_rate)
			theta = np.subtract(theta,value) # Updating theta
		e.append(self.error(x,y,theta))
		return theta,e

	def fit(self, x, y, m="bgd"): # Fit function. Parameters - Numpy input array, Numpy output array, string to determine sgd or bgd use
		x, y = x[:], y[:]
		x = np.concatenate((x,np.ones((len(x),1))),axis=1) # Adding a column of 1s
		y = np.reshape(y,(-1,1)) # Conversion of 1D array to 2D

		if m.lower()=="bgd": # If user wants BGD
			self.theta, self.e = self.fitBGD(x, y)
		else: # In any other case
			self.theta, self.e = self.fitSGD(x, y)
		# print (self.theta)
		return self

	def predict(self, X): # Predicting output values for a given X and on final parameters.
		X = X[:] # copying the array
		X = np.concatenate((X,np.ones((len(X),1))),axis=1) # Adding a column of 1s
		y = self.hypothesis(X,self.theta) # Calculates the probabilities for that datapoint
		ans = []
		for i in y:
			if i>=0.5: # If the probability is greater than or equal to 0.5, then final values is taken to be one.
				ans.append(1)
			else:
				ans.append(0)
		return ans

	def accuracy(self, x, y): # Takes input array and actual y as parameters. Predicted y is first calcuated for input x and then accuracy is calculated.
		x, y = x[:], y[:] # Copying the array
		x = np.concatenate((x,np.ones((len(x),1))),axis=1) # Adding a column of 1s
		y = np.reshape(y,(-1,1)) # Conversion of 1D array to 2D

		correct = 0
		for i in range(len(y)):
			v = self.hypothesis(x[i],self.theta) # Calculating Probability for that datapoint
			if v<0.5:
				if y[i]==0:
					correct += 1
			elif v>=0.5:
				if y[i]:
					correct += 1

		return correct/len(y)

	def plot(self): # Plotting the graph
		x,y1,y2 = [],[],[]
		for i in range(1,len(self.dtrain)+1):
			x.append(i)
			y1.append(self.dtrain[i])
			y2.append(self.dtest[i])
		plt.plot(x,y1,label="Training")
		plt.plot(x,y2,label="Validation")
		plt.legend()
		plt.show()


"""
BELOW CODE PLOTS THE GRAPH FOR ALL THE FOLDS FOR A PARTICULAR
VALUE OF K FOR THE GIVEN DATASETS AND FOR A PARTICULAR ERROR.
IF DIFFERENT ERROR IS REQUIRED, CHANGE THE FUNCTION CALL AND
SIMILARLY IF DIFFERENT K IS REQUIRED CHANGE THE VALUE WHICH
IS ASSIGNED TO K.
"""

# p = MyPreProcessor()
# model = MyLinearRegression()
# for data in range(1):
# 	x,y = p.pre_process(data)
# 	k = 10
# 	for fold in range(k):
# 		length = len(y)//k
# 		xtrain = np.array(x[0:length*fold]+x[length*(fold+1):])
# 		xtest = np.array(x[length*fold:length*(fold+1)])
# 		ytrain = np.array(y[0:length*fold]+y[length*(fold+1):])
# 		ytest = np.array(y[length*fold:length*(fold+1)])


# 		xtest = np.concatenate((xtest,np.ones((len(xtest),1))),axis=1) # Adding a column of 1s
# 		ytest = np.reshape(ytest,(-1,1)) # Conversion of 1D array to 2D
# 		model.fit(xtrain,ytrain,"rmse")

# 		print ("MAE error on testing set : ",model.errorRMSE(xtest,ytest,model.theta))
# 		model.plot()



"""
BELOW k_fold() FUNCTION PRINTS THE RMSE AND MAE VALUES FOR 
ALL THE VALUES OF K FROM 3 TO 10 AND FOR BOTH THE DATASETS.
IT WILL TAKE AROUND 40 SECONDS TO COMPUTE THESE 32 VALUES.
"""

def k_fold():
	p = MyPreProcessor()
	model = MyLinearRegression()
	loss = {}
	start = time.time() # Calculating The time
	for k in range(3,11): # Calculating for All Folds
		loss[k] = [0,0]
		for data in range(2): # For both datasets
			x,y = p.pre_process(data)
			rmse,mae = 0,0
			t1,t2 = 0,0
			for fold in range(k): # For k folds in k-fold
				length = len(y)//k
				xtrain = np.array(x[0:length*fold]+x[length*(fold+1):])
				xtest = np.array(x[length*fold:length*(fold+1)])
				ytrain = np.array(y[0:length*fold]+y[length*(fold+1):])
				ytest = np.array(y[length*fold:length*(fold+1)])

				model.fit(xtrain,ytrain,"rmse") # Train model using RMSE
				predictedRMSE = model.predict(xtest)
				t1 = model.errorRMSE(np.concatenate((xtest,np.ones((len(xtest),1))),axis=1),np.reshape(ytest,(-1,1)),model.theta)
				rmse += t1

				model.fit(xtrain,ytrain,"mae") # Train model using mae
				predictedMAE = model.predict(xtest)
				t2 = model.errorMAE(np.concatenate((xtest,np.ones((len(xtest),1))),axis=1),np.reshape(ytest,(-1,1)),model.theta)
				mae += t2
				# print ("RMSE Value for Fold ",fold+1," : ",t1)
				# print ("MAE : ",t2)

			# print ("RMSE Error on dataset ",data," using ",k," folds : ",rmse/k)
			# print ("MAE Error on dataset ",data," using ",k," folds : ",mae/k)
			loss[k][data] = [rmse/k,mae/k]

	print ("TIME TAKEN BY THIS MODEL : ",time.time()-start) # Total time for the model

	for data in range(2):
		for k in range(3,11):
			# print ("--------------------------------------------------------------")
			# print ("RMSE Error on dataset ",0," using ",k," folds : ",loss[k][0][0])
			# print ("MAE Error on dataset ",0," using ",k," folds : ",loss[k][0][1])
			print ("RMSE Value on dataset ",data+1," using ",k," folds : ",loss[k][data][0])
		for k in range(3,11):
			print ("MAE Value on dataset ",data+1," using ",k," folds : ",loss[k][data][1])

"""
BELOW optimal_param() FUNCTION PRINTS THE PARAMETERS OBTAINED
NORMAL EQUATION FORM FUNCTION FOR DATASET 1 AND FOR A PARTICULAR
VALUE OF K AND FOR THE BEST FOLD AND FOR THE BEST ERROR.
"""

def optimal_param():
	p = MyPreProcessor()
	model = MyLinearRegression()
	dataset = 0
	k = 10 # For a particular k
	best_fold = 4 # Best fold for dataset 1 and k = 10
	x,y = p.pre_process(0)
	length = len(x)//k
	xtrain = np.array(x[0:(best_fold-1)*length]+x[best_fold*length:])
	ytrain = np.array(y[0:(best_fold-1)*length]+y[best_fold*length:])
	xtest = np.array(x[(best_fold-1)*length:best_fold*length])
	ytest = np.array(y[(best_fold-1)*length:best_fold*length])
	# print (len(xtrain),len(xtrain[0]),len(ytrain),len(xtest),len(xtest[0]))
	xtrain = np.concatenate((xtrain,np.ones((len(xtrain),1))),axis=1) # Adding a column of 1s to X array. Example - "To take care of c in case of y = mx + c"
	ytrain = np.reshape(ytrain,(-1,1)) # Conversion of 1D array to 2D
	xtest = np.concatenate((xtest,np.ones((len(xtest),1))),axis=1) # Adding a column of 1s to X array. Example - "To take care of c in case of y = mx + c"
	ytest = np.reshape(ytest,(-1,1)) # Conversion of 1D array to 2D
	xtrain = (xtrain - xtrain.mean())/(xtrain.std()) # Normalise
	param = model.normal_equation(xtrain,ytrain)


	print ("RMSE value for training set using optimal parameters : ",model.errorRMSE(xtrain,ytrain,param))
	print ("RMSE value for validation set using optimal parameters : ",model.errorRMSE(xtest,ytest,param))

'''
UNCOMMENT BELOW FUNCTION CALLS TO RUN CODE
'''
# k_fold()
# optimal_param()


"""
THE FOLLOWING CODE USES LOGISTIC REGRESSION AND SPLITS
THE DATA INTO TRAINING SET, VALIDATION SET AND TESTING 
SET IN THE RATIO OF 7:1:2. THE DATA IS TRAINED ON THE 
TRAINING SET GIVES THE ACCURACY ON BOTH TRAINING SET
ANS TESTING SET. IT ALSO PLOTS THE ERROR VS ITERATION 
GRAPH FOR BOTH TRAINING SET AND VALIDATION SET.
"""

# p = MyPreProcessor()
# model = MyLogisticRegression()
# x,y = p.pre_process(2)

# # Splitting in the ratio of 7:1:2 and converting to numpy array.
# xtrain = np.array(x[0:7*(len(x)//10)])
# ytrain = np.array(y[0:7*(len(y)//10)])
# xval = np.array(x[7*(len(x)//10):8*(len(x)//10)])
# yval = np.array(y[7*(len(y)//10):8*(len(y)//10)])
# xtest = np.array(x[8*(len(x)//10):])
# ytest = np.array(y[8*(len(y)//10):])

# xval = np.concatenate((xval,np.ones((len(xval),1))),axis=1) # Adding a column of 1s
# yval = np.reshape(yval,(-1,1)) # Conversion of 1D array to 2D
# model.fit(xtrain,ytrain,"bgd") # Train using BGD
# print ("Accuracy on Training Set using BGD : ",model.accuracy(xtrain,ytrain))
# print ("Accuracy on Testing Set using BGD : ",model.accuracy(xtest,ytest))
# model.plot()
# model.fit(xtrain,ytrain,"sgd") # Train using SGD
# print ("Accuracy on Training Set using SGD : ",model.accuracy(xtrain,ytrain))
# print ("Accuracy on Testing Set using SGD : ",model.accuracy(xtest,ytest))
# model.plot() # Plotting the graph


