from tensorflow import set_random_seed
from keras import models, layers
import numpy as np
import sincnet
from keras.layers import Dense, Dropout, Activation
from keras.layers import MaxPooling1D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten
from keras.layers import InputLayer, Input
from keras.models import Model

from config import *

def get_model(input_shape, out_dim):
    inputs = Input(input_shape)

    x = sincnet.SincConv1D(layer_1_cnn_n_filters, layer_1_cnn_len_filter, sr)(inputs)
    x = MaxPooling1D(pool_size=layer_1_cnn_maxpool_size)(x)
    x = BatchNormalization(momentum=0.05)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(layer_2_dnn_n)(x)
    x = BatchNormalization(momentum=0.05)(x)
    x = LeakyReLU(alpha=0.2)(x)

    prediction = layers.Dense(out_dim, Activation="softmax")(x)

    model = Model(inputs = inputs, outputs = prediction)
    model.summary()
    return model
