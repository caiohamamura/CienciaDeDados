import pandas as pd

# ANN hyperparams
inputNumber = 5
layer1Neurons = 5
learningRate = 0.01
momentumFactor = 0.8
precision = 0.5e-6

# Concatenate all data
dataSet = pd.read_csv("datatraining.txt")
dataSet = pd.concat([dataSet, pd.read_csv("datatest.txt")])
validationSet = pd.read_csv("datatest2.txt")


# Drop date column
dataSet.drop("date", 1, inplace=True)
validationSet.drop("date", 1, inplace=True)

# Separate data (X) and response (y) vectors
X = dataSet.iloc[:,:-1]
y = dataSet.iloc[:,-1]
validation_X = validationSet.iloc[:,:-1]
validation_y = validationSet.iloc[:,-1]


# Normalize to [0-1]
minVals = X.min()
maxVals =  X.max()
X = ((X-minVals)/(maxVals-minVals))


# 

trainingSet.shape

