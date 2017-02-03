import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD


# FIX
import tensorflow as tf
tf.python.control_flow_ops = tf
# END fix

import glob
files =  glob.glob("../build/data/data*.npy")

bigdata = []
bigresp = []

for f in files:
    id = int(filter(str.isdigit, f))
    data = np.asarray(np.load(f))
    resp = np.asarray(np.load('../build/data/resp_'+str(id)+'.npy'))
    # TODO:: Change input shape before adding to channels, x ,y
    bigdata.append(data.reshape(8, 32, 32))
    bigresp.append(resp)

bigdata = np.asarray(bigdata)
bigresp = np.asarray(bigresp)

print 'Data loaded.'

print 'Data: ' + str(bigdata.shape)
print 'Resp: ' + str(bigresp.shape)

for x in xrange(1,7):
    print('Max data: ' + str(np.max(bigresp[x])))
    print('Min data: ' + str(np.min(bigresp[x])))

import matplotlib.pyplot as plt

plt.hist(bigresp, bins=20)  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram with 20 bins")
plt.show()

# print 'shuffling data'
# from random import shuffle, seed
# index_shuf = range(len(bigdata))
# seed(0)
# shuffle(index_shuf)
# bigdata = [bigdata[i] for i in index_shuf]
# bigresp = [bigresp[i] for i in index_shuf]
# print 'Done shuffling.'


def create_model(weights_path=None):
    with tf.device('/cpu:0'):
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(8, 32, 32))) ## CHANGE INPUT SHAPE!!
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        # model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        # model.add(ZeroPadding2D((1,1)))
        # model.add(Convolution2D(512, 3, 3, activation='relu'))
        # model.add(ZeroPadding2D((1,1)))
        # model.add(Convolution2D(512, 3, 3, activation='relu'))
        # model.add(ZeroPadding2D((1,1)))
        # model.add(Convolution2D(512, 3, 3, activation='relu'))
        # model.add(MaxPooling2D((2,2), strides=(2,2)))

        # model.add(ZeroPadding2D((1,1)))
        # model.add(Convolution2D(512, 3, 3, activation='relu'))
        # model.add(ZeroPadding2D((1,1)))
        # model.add(Convolution2D(512, 3, 3, activation='relu'))
        # model.add(ZeroPadding2D((1,1)))
        # model.add(Convolution2D(512, 3, 3, activation='relu'))
        # model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='linear'))

    if weights_path:
        model.load_weights(weights_path)

    return model

# Generate model
batch_size = 512
# latent_dim = len(classes)
nb_epoch = 10

x_train = np.asarray(bigdata)
y_train = np.asarray(bigresp)


inps = create_model() # or h5 for storing the model
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
inps.compile(optimizer=sgd, loss='mse')

hist = inps.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)
inps.save('inps.h5')  # creates a HDF5 file

# loss_and_metrics = inps.evaluate(x_test, y_test, batch_size=batch_size)
# print loss_and_metrics
