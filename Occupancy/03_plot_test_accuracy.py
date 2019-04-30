# MLP for Pima Indians Dataset Serialize to JSON and HDF5
import pandas as pd
import numpy as np

# import models from networks_definition
from networks_definition import models

# Load training result
trainResult = pd.read_csv("data/trainResult.csv")

#Load test set
testSet = pd.read_csv("data/normalizedTest.csv")
X=testSet.iloc[:,:-1]
y=testSet.iloc[:,-1]


# Dataframe to store f1-scores from testing
testResult = pd.DataFrame({
    "network":[],
    "iter":[],
    "accuracy":[],
    "f1-score":[]
})

# For each network and iter: we have 6 types of networks
# and 10 iterations, each corresponding to a fold of 
# K-fold, where K=10
from sklearn import metrics
for network in range(6):
    #Get as roman
    ann_roman = {
        1:"I",
        2:"II",
        3:"III",
        4:"IV",
        5:"V",
        6:"VI"
    }[network+1]
    for iter in range(10):
        # Load weights stores in data folder
        models[network].load_weights("data/model-%d-iter-%d.hdf5" % (network+1, iter+1))
        model = models[network]

        # Calculate goodness
        f1_score=metrics.f1_score(y, model.predict_classes(X))
        accuracy=metrics.accuracy_score(y, model.predict_classes(X))

        # Save results
        testResult.loc[testResult.shape[0]] = [ann_roman, iter+1, accuracy, f1_score]
        
# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Figure resolution
plt.figure(dpi=150)
# White style
sns.set_style("whitegrid") 
# First color from muted pallete
color=sns.color_palette("muted")[0] 

# Seaborn boxplot with fixed color
sns.boxplot(x="network", y="f1-score", color=color, data=testResult)
plt.xlabel("Network")
plt.ylabel("Test F1-score")
plt.show()
