from model import *
from config import *
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

model = get_model(input_shape, output_dim)
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
