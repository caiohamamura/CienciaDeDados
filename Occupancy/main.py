import pandas as pd
# Print legend outside jupyter
# %matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random 

# ANN hyperparams
inputNumber = 5
layer1Neurons = 5
learningRate = 0.3
momentumFactor = 0.8
seed=424785
kFold = 10
repeatKFold = 1
batchSize = 200
maxEpoch = 30
improveTol = maxEpoch

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



# MLP Classification logistic
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import numpy as np

splitter = StratifiedKFold(n_splits=kFold, random_state=seed)

trainResult = pd.DataFrame({
    "network":np.array([], dtype=np.object), 
    "iter":np.array([], dtype=np.int), 
    "epoch":np.array([], dtype=np.int), 
    "type":np.array([], dtype=np.object), 
    "loss": np.array([], dtype=np.float64), 
    "accuracy": np.array([], dtype=np.float64)})


configs = [{ #NETWORK I
    "solver":'sgd',
    "activation":'logistic', 
    "learning_rate_init":learningRate,
    "learning_rate":"adaptive",
    "batch_size":batchSize,
    "hidden_layer_sizes":(
        5
    ), 
    "random_state":seed,
    "shuffle":True
}, { #NETWORK II
    "solver":'sgd',
    "activation":'logistic', 
    "learning_rate_init":learningRate,
    "learning_rate":"adaptive",
    "batch_size":batchSize,
    "hidden_layer_sizes":(
        20
    ), 
    "random_state":seed,
    "shuffle":True
}, { #NETWORK III
    "solver":'sgd',
    "activation":'logistic', 
    "learning_rate_init":learningRate,
    "learning_rate":"adaptive",
    "batch_size":batchSize,
    "hidden_layer_sizes":(
        5,
        5,
    ), 
    "random_state":seed,
    "shuffle":True
}, { #NETWORK IV
    "solver":'sgd',
    "activation":'tanh', 
    "learning_rate_init":learningRate,
    "learning_rate":"adaptive",
    "batch_size":batchSize,
    "hidden_layer_sizes":(
        5
    ), 
    "random_state":seed,
    "shuffle":True
}, { #NETWORK V
    "solver":'sgd',
    "activation":'relu', 
    "learning_rate_init":learningRate,
    "learning_rate":"adaptive",
    "batch_size":batchSize,
    "hidden_layer_sizes":(
        5
    ), 
    "random_state":seed,
    "shuffle":True
}
]

ann_index = 1
# Test each network configuration
for config in configs:
    # Convert to roman numeral
    ann_roman = {
        1:"I",
        2:"II",
        3:"III",
        4:"IV",
        5:"V"
    }[ann_index]
    print()
    print("-"*32)
    print("NETWORK %s" % ann_roman)
    print("-"*32)
    iter = 1

    # For each split of KFolding
    for train_index, validation_index in splitter.split(X, y):
        clf = MLPClassifier(**config)
        X_validation, y_validation = X.iloc[validation_index], y.iloc[validation_index]
        best = 0
        bestTrain = 0
        bestModel = []
        epoch = 1
        lastImprove = 0
        print("Iteration %d/%d" % (iter, kFold*repeatKFold))
        while (True):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            currentModel = clf.partial_fit(X_train, y_train, classes=[0,1])
            if lastImprove == improveTol or epoch > maxEpoch: break   
            current = clf.score(X_train, y_train)
            trainLoss = log_loss(clf.predict(X_train), y_train, labels=[0,1])
            currentCross = clf.score(X_validation, y_validation)
            valLoss = log_loss(clf.predict(X_validation), y_validation, labels=[0,1])
            trainResult.loc[trainResult.shape[0]] = [ann_roman, iter, epoch, "train", trainLoss, current]
            trainResult.loc[trainResult.shape[0]] = [ann_roman, iter, epoch, "validation", valLoss, currentCross]
            if current > bestTrain:
                bestTrain = current
                lastImprove = 0
            if currentCross > best:
                best = currentCross
                bestModel = currentModel
                lastImprove = 0
            bar="=" * int(round(32*epoch/maxEpoch)) + " "*int(round((32*(maxEpoch-epoch)/maxEpoch)))
            print("\r[%s] epoch %d/%d - loss: %.4f, acc: %.4f, val_loss: %.4f, val_acc: %.4f" % (bar, epoch, maxEpoch, trainLoss, current, valLoss, currentCross), end="")
            epoch += 1
            lastImprove += 1
            random.shuffle(train_index)
        iter += 1
        print("")
    ann_index += 1

# Plot style
sns.set_style("whitegrid") 
palette=sns.color_palette("muted")[0:trainResult["network"].unique().size] 


# Plot
plt.figure(dpi=150)
ax = sns.lineplot(x="epoch", y="accuracy", hue="network", style="type", data=trainResult, palette=palette)


# Plot configs
params = {"loc":"lower right"}
leg = ax.legend()
handles = leg.legendHandles # Remove handler for type

handles[0].set_label("ANN")
handles[6].set_label("Set")
ax.legend(handles=handles, **params)
ax.set_xlabel(ax.get_xlabel().capitalize())
ax.set_ylabel(ax.get_ylabel().capitalize())
plt.xticks(pd.np.arange(0, trainResult["epoch"].max()+1, 2))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

from radarboxplot import radarboxplot
plt.figure(dpi=150)
axs=radarboxplot(X, y, X.columns.values, nrows=1, ncols=2)
plt.show()