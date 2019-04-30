# Needs pip install h5py

import pandas as pd
# Print legend outside jupyter
# %matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random 
import numpy as np
from keras.layers import Layer
from keras import backend as K
from keras import layers
import keras
from tensorflow import set_random_seed
import json

# Create RBF Keras Layer
class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)



# ANN hyperparams
inputNumber = 5
learningRate = 0.3
momentumFactor = 0.8
seed=424785
kFold = 10
repeatKFold = 1
batchSize = 200
maxEpoch = 50
improveTol = maxEpoch
np.random.seed(seed)
set_random_seed(seed)


# Concatenate all data
dataSet = pd.read_csv("datatraining.txt")
dataSet = pd.concat([dataSet, pd.read_csv("datatest.txt")])
testSet = pd.read_csv("datatest2.txt")


# Drop date column
dataSet.drop("date", 1, inplace=True)
testSet.drop("date", 1, inplace=True)

# Separate data (X) and response (y) vectors
dataSet.to_csv("data/normalizedData.csv", index=False)
X = dataSet.iloc[:,:-1]
y = dataSet.iloc[:,-1]
test_X = testSet.iloc[:,:-1]
test_y = testSet.iloc[:,-1]


# Normalize to [0-1]
minVals = X.min()
maxVals =  X.max()
X = ((X-minVals)/(maxVals-minVals))



# MLP Classification logistic
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

splitter = StratifiedKFold(n_splits=kFold, random_state=seed)

trainResult = pd.DataFrame({
    "network":np.array([], dtype=np.object), 
    "iter":np.array([], dtype=np.int), 
    "epoch":np.array([], dtype=np.int), 
    "type":np.array([], dtype=np.object), 
    "loss": np.array([], dtype=np.float64), 
    "accuracy": np.array([], dtype=np.float64)})


models = [
    #NETWORK I
    keras.models.Sequential([
        layers.Dense(5, input_shape=(5,), activation="sigmoid"),
        layers.Dense(1, activation="sigmoid")
    ]),
    #NETWORK II
    keras.models.Sequential([
        layers.Dense(10, input_shape=(5,), activation="sigmoid"),
        layers.Dense(1, activation="sigmoid")
    ]),
    #NETWORK III
    keras.models.Sequential([
        layers.Dense(15, input_shape=(5,), activation="sigmoid"),
        layers.Dense(1, activation="sigmoid")
    ]),
    #NETWORK IV
    keras.models.Sequential([
        layers.Dense(5, input_shape=(5,), activation="sigmoid"),
        layers.Dense(1, activation="sigmoid")
    ]),
    #NETWORK V
    keras.models.Sequential([
        RBFLayer(5, 0.5, input_shape=(5,)),
        layers.Dense(1, activation="sigmoid")
    ]),
    #NETWORK VI
    keras.models.Sequential([
        layers.Dense(5, input_shape=(5,), activation="sigmoid"),
        layers.Dense(5, activation="sigmoid"),
        layers.Dense(1, activation="sigmoid")
    ]) 
]

for m in models: 
    m.compile(optimizer=keras.optimizers.SGD(lr=learningRate),
              loss="binary_crossentropy",
              metrics=["accuracy"])

models[3].compile(
    optimizer=keras.optimizers.SGD(
        lr=learningRate, 
        momentum=momentumFactor
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"])


ann_index = 1
# Test each network configuration
for model in models:
    # Convert to roman numeral
    ann_roman = {
        1:"I",
        2:"II",
        3:"III",
        4:"IV",
        5:"V",
        6:"VI"
    }[ann_index]
    print()
    print("-"*32)
    print("NETWORK %s" % ann_roman)
    print("-"*32)
    iter = 1

    # For each split of KFolding
    for train_index, validation_index in splitter.split(X, y):
        X_validation, y_validation = X.iloc[validation_index], y.iloc[validation_index]
        print("Iteration %d/%d" % (iter, kFold*repeatKFold))
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        fitted=model.fit(
            x=X_train,
            y=y_train,
            batch_size=batchSize,
            epochs=50,
            verbose=1,
            validation_data=(X_validation, y_validation)
        )
        data=pd.DataFrame(fitted.history)
        
        trainResult = trainResult.append(pd.DataFrame({
            "network": [ann_roman]*data.shape[0],
            "iter": [iter]*data.shape[0],
            "epoch": data.index.values+1,
            "type": ["train"]*data.shape[0],
            "loss":data["loss"],
            "accuracy":data["acc"]
        }), ignore_index=True)

        trainResult = trainResult.append(pd.DataFrame({
            "network": [ann_roman]*data.shape[0],
            "iter": [iter]*data.shape[0],
            "epoch": data.index.values+1,
            "type": ["validation"]*data.shape[0],
            "loss":data["val_loss"],
            "accuracy":data["val_acc"]
        }), ignore_index=True)

        model_json = model.to_json()
        with open("data/model%s.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

        iter += 1
        print("")
    ann_index += 1



trainResult.to_csv("data/trainResult.csv")