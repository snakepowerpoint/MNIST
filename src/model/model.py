# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:18:02 2018

@author: RahulJY_Wang
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from sklearn.svm import LinearSVC

from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.layers import Dropout, Flatten
from keras.models import Model
from keras.utils import layer_utils
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.regularizers import l1_l2

def svm_sklearn(X, y):

    clf = LinearSVC(random_state=0)
    clf.fit(X, y)
    score = clf.decision_function(X)
    prediction = clf.predict(X)

    score_pd = pd.DataFrame(score)
    score_sort = np.sort(score, axis=1)
    score_sort = np.argsort(score, axis=1)

    score_sort = np.argsort(score)


def nn_keras(input_shape, num_class = None):

    X_input = Input(input_shape)
    X = Dense(16, kernel_initializer='he_uniform', name='fc1')(X_input)
    X = Activation('relu')(X)
    X = BatchNormalization(name='bn1')(X)

    X = Dense(32, kernel_initializer='he_uniform', name='fc2')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(name='bn2')(X)

    X = Dense(num_class, activation='softmax', kernel_initializer='he_uniform', name='output')(X)

    model = Model(inputs=X_input, outputs=X, name='nn_keras')

    return model


"""
def nn_keras_s():
"""
