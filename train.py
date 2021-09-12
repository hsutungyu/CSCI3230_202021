import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# setup training data
# using pickle package to unpack .pickle file provided
infile = open('../SR-ARE-train/names_onehots.pickle', 'rb')
rawdict = pickle.load(infile)
infile.close()

# separating dict into onehots and names
train_onehots = rawdict["onehots"]
# add channels dimension
train_onehots = np.expand_dims(train_onehots, -1)
train_names = rawdict["names"]

# loading labels from names_labels.txt
train_labels = []
with open('../SR-ARE-train/names_labels.txt', 'r') as filestream:
    for line in filestream:
        currentline = line.rstrip().split(',')
        # then currentline[1] is the label
        train_labels.append(currentline[1])

# change labels to int
train_labels = list(map(int, train_labels))

# since the training set is unbalanced, use the method of custom weights
# 1. separate training set into positive and negative samples
train_pos_onehots = []
train_neg_onehots = []
train_pos_labels = []
train_neg_labels = []
for x in range(len(train_labels)):
    if train_labels[x] == 1:
        train_pos_onehots.append(train_onehots[x])
        train_pos_labels.append(1)
    else:
        train_neg_onehots.append(train_onehots[x])
        train_neg_labels.append(0)

# 2. find custom weight
neg = len(train_neg_onehots)
pos = len(train_pos_onehots)
weight_for_0 = (1 / neg) * (neg + pos) / 2.0
weight_for_1 = (1 / pos) * (neg + pos) / 2.0
class_weight = {0: weight_for_0, 1: weight_for_1}

# find number of rows and columns of one-hot representation
train_rows = train_onehots.shape[1]
train_columns = train_onehots.shape[2]

# setup testing data
# using pickle package to unpack .pickle file provided
infile = open('../SR-ARE-test/names_onehots.pickle', 'rb')
rawdict = pickle.load(infile)
infile.close()

# separating dict into onehots and names
test_onehots = rawdict["onehots"]
# add channels dimension
test_onehots = np.expand_dims(test_onehots, -1)
test_names = rawdict["names"]

# loading labels from names_labels.txt
test_labels = []
with open('../SR-ARE-test/names_labels.txt', 'r') as filestream:
    for line in filestream:
        currentline = line.rstrip().split(',')
        # then currentline[1] is the label
        test_labels.append(currentline[1])

# change labels to int
test_labels = list(map(int, test_labels))
# find number of rows and columns of one-hot representation
test_rows = test_onehots.shape[1]
test_columns = test_onehots.shape[2]

# set initial bias
initial_bias = np.log([pos / neg])

# model creation
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(int(train_rows), int(train_columns), int(1)), kernel_regularizer='l2',
                        kernel_initializer=tf.keras.initializers.RandomUniform()))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(initial_bias)))

# optimizer: SGD
opt = optimizers.SGD(lr=0.01)

# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='AUC')])

# early stopping, no improvement on val_acc for 3 epochs = stop
callbackEarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
callbackSaveBestAUC = tf.keras.callbacks.ModelCheckpoint(filepath='my_model/bestAUC.h5', monitor='val_AUC',
                                                         mode='max', save_best_only=True)

# train model
history = model.fit(train_onehots, train_labels, epochs=500,
                    validation_data=(test_onehots, test_labels), batch_size=256,
                    callbacks=[callbackSaveBestAUC], class_weight=class_weight)

# save model
if not os.path.exists('my_model'):
    os.mkdir('my_model')
model.save('my_model/my_model.h5')
