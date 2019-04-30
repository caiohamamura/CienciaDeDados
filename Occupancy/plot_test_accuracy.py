# MLP for Pima Indians Dataset Serialize to JSON and HDF5
import pandas as pd
import numpy as np
from keras.layers import Layer
import keras
from keras import layers
from keras import backend as K

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




# Load training result
trainResult = pd.read_csv("data/trainResult.csv")


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
        RBFLayer(2, 0.5, input_shape=(5,)),
        layers.Dense(1, activation="sigmoid")
    ]),
    #NETWORK VI
    keras.models.Sequential([
        layers.Dense(5, input_shape=(5,), activation="sigmoid"),
        layers.Dense(5, activation="sigmoid"),
        layers.Dense(1, activation="sigmoid")
    ]) 
]

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
