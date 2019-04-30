# Needs pip install h5py

import pandas as pd
# Print legend outside jupyter
# %matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random 
import numpy as np
from keras import backend as K
from keras import layers
import keras
from tensorflow import set_random_seed
from keras.callbacks import ModelCheckpoint

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
X = dataSet.iloc[:,:-1]
y = dataSet.iloc[:,-1]
test_X = testSet.iloc[:,:-1]
test_y = testSet.iloc[:,-1]


# Normalize to [0-1]
minVals = X.min()
maxVals =  X.max()
X = ((X-minVals)/(maxVals-minVals))
dataSet.iloc[:,:-1] = X

# Normalize test set
minVals = test_X.min()
maxVals =  test_X.max()
test_X = ((test_X-minVals)/(maxVals-minVals))
testSet.iloc[:,:-1] = test_X

# Save normalized sets
dataSet.to_csv("data/normalizedData.csv", index=False)
dataSet.to_csv("data/normalizedTest.csv", index=False)


# MLP Classification logistic
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

# Use stratified k-fold
splitter = StratifiedKFold(n_splits=kFold, random_state=seed)

# Dataset for storing results from training
trainResult = pd.DataFrame({
    "network":np.array([], dtype=np.object), 
    "iter":np.array([], dtype=np.int), 
    "epoch":np.array([], dtype=np.int), 
    "type":np.array([], dtype=np.object), 
    "loss": np.array([], dtype=np.float64), 
    "accuracy": np.array([], dtype=np.float64)})


from networks_definition import models

pd.to_pickle(models, "data/models.pkl")


# Reset weights for model
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

# Compile models
for m in models: 
    m.compile(optimizer=keras.optimizers.SGD(lr=learningRate),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Network 4 uses momentum
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


        # Monitor trainning to save best model
        filepath="data/model-%d-iter-%d.hdf5" % (ann_index, iter)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        reset_weights(model)

        # Fit model
        fitted=model.fit(
            x=X_train,
            y=y_train,
            batch_size=batchSize,
            epochs=50,
            verbose=0,
            validation_data=(X_validation, y_validation),
            callbacks=callbacks_list
        )

        # Save accuracy loss to trainResult
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

        iter += 1
        print("")
    ann_index += 1


# Save training
trainResult.to_csv("data/trainResult.csv", index=False)