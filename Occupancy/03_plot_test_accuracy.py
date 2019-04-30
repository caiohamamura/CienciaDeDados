# MLP for Pima Indians Dataset Serialize to JSON and HDF5
import pandas as pd
import numpy as np

import keras
from keras import layers
from keras import backend as K


# import models from networks_definition
from networks_definition import models

# Load training result
trainResult = pd.read_csv("data/trainResult.csv")



#Load test set
from sklearn import metrics
testSet = pd.read_csv("data/normalizedTest.csv")
X=testSet.iloc[:,:-1]
y=testSet.iloc[:,-1]



# Load weights
for network in range(6):
    ann_roman = {
        1:"I",
        2:"II",
        3:"III",
        4:"IV",
        5:"V",
        6:"VI"
    }[network+1]
    for iter in range(10):
        models[network].load_weights("data/model-%d-iter-%d.hdf5" % (network+1, iter+1))
        model = models[network]
        accuracy=metrics.f1_score(y, model.predict_classes(X))
        logLoss=metrics.log_loss(y, model.predict_classes(X))
        trainResult.loc[trainResult.shape[0]] = [ann_roman, iter+1, 0, "test", logLoss, accuracy]
        
validation=trainResult[trainResult["type"]=="validation"].groupby(["network", "iter"])["accuracy"].agg({"max":np.max})
test=trainResult[trainResult["type"]=="test"].groupby(["network", "iter"])["accuracy"].agg({"maxTest":np.max})
valid_test = validation.join(test)
valid_test.reset_index(level=0, inplace=True)

#%matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(dpi=150)
palette=sns.color_palette("muted")[0:trainResult["network"].unique().size] 
sns.set_style("whitegrid") 
sns.boxplot(x="network", y="maxTest", color=palette[0], data=valid_test)
plt.xlabel("Network")
plt.ylabel("Test F1-Score")
plt.show()
