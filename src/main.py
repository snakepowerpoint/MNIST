import sys
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

from model import model

import keras
import keras.backend as K
'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
'''

# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

y = pd.DataFrame({'label':y})
y = pd.get_dummies(y['label'])


if __name__ == '__main__':
    
    # Keras DNN
    if sys.argv[1] == 'keras':
        print('Apply deep neural network using Keras \n')
        K_dnn = model.nn_keras(X.shape[1:], num_class=3)
        adam = keras.optimizers.Adam(lr=0.01, decay=1e-6)

        K_dnn.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        K_dnn.fit(X, y, epochs=100, batch_size=32)




    # Tensorflow DNN
