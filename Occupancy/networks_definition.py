import keras
from keras import layers
from keras import backend as K
from keras.layers import Layer

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