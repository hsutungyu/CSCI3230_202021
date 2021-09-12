import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pickle
import os

# model recreate
model = tf.keras.models.load_model('trained_model.h5')

# setup testing data
# using pickle package to unpack .pickle file provided
infile = open('../SR-ARE-score/names_onehots.pickle', 'rb')
rawdict = pickle.load(infile)
infile.close()

# separating dict into onehots and names
test_onehots = rawdict["onehots"]
# add channels dimension
test_onehots = np.expand_dims(test_onehots, -1)


# binary classification, output to text file
prediction = model.predict(test_onehots)
with open('labels.txt', 'w') as f:
    for item in prediction:
        if item >= 0.5:
            f.write("1\n")
        else:
            f.write("0\n")

