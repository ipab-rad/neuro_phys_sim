# import cv2
import numpy as np
import time as t
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, merge
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, Adamax

# MPL no display backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# end

# FIX
import tensorflow as tf
tf.python.control_flow_ops = tf
# END fix

import re
import glob
# files =  glob.glob("../build/data/data*.npy")

# bigdata = []
# bigresp = []

# Get params
min_data2 = np.asarray([-6.64888525, -13.71338463])
max_data = np.asarray([12.40662193, 19.73525047])
mean_data = np.load('../src/models/mean_data2full.npy')
mean_data = mean_data.reshape((1, 32, 32, 8))
print('Min data:  ' + str(min_data2))
print('Max data:  ' + str(max_data))
print('Mean data:  ' + str(mean_data.shape))
# print('Mean data:  ' + str(mean_data))

# Shift input mean data
# print('Mean input data: ' + str(np.mean(bigdata, axis = 0)))
# np.mean(bigdata, axis = 0).tofile('mean_data2full.dat')
# print('Input data shifted')

print('Data loaded.')

# Generate model
batch_size = 256*4
# latent_dim = len(classes)
# nb_epoch = 250 #250*4
# latent_dim = 64


# x_train = np.asarray(bigdata)
# y_train = np.asarray(bigresp)

from keras import initializations
def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def create_model(weights_path=None):
    with tf.device('/cpu:0'):
        obj_inp0 = Input(batch_shape=(None, 32, 32, 1), name='obj_world')
        x0 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(obj_inp0)
        # x0 = ZeroPadding2D((1,1))(x0)
        # x0 = MaxPooling2D((2,2), strides=(2,2))(x0)
        x0 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(x0)
        x0 = MaxPooling2D((2,2), strides=(2,2))(x0)
        x0 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x0)
        x0 = MaxPooling2D((2,2), strides=(2,2))(x0)
        x0 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x0)
        x0 = MaxPooling2D((2,2), strides=(2,2))(x0)
        conv_out0 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x0)

        obj_inp1 = Input(batch_shape=(None, 32, 32, 1), name='obj_identity')
        x1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(obj_inp1)
        # x1 = ZeroPadding2D((1,1))(x1)
        # x1 = MaxPooling2D((2,2), strides=(2,2))(x1)
        x1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(x1)
        x1 = MaxPooling2D((2,2), strides=(2,2))(x1)
        x1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x1)
        x1 = MaxPooling2D((2,2), strides=(2,2))(x1)
        x1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x1)
        x1 = MaxPooling2D((2,2), strides=(2,2))(x1)
        conv_out1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x1)

        obj_inp2 = Input(batch_shape=(None, 32, 32, 1), name='mass')
        x2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(obj_inp2)
        # x2 = ZeroPadding2D((1,1))(x2)
        # x2 = MaxPooling2D((2,2), strides=(2,2))(x2)
        x2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(x2)
        x2 = MaxPooling2D((2,2), strides=(2,2))(x2)
        x2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x2)
        x2 = MaxPooling2D((2,2), strides=(2,2))(x2)
        x2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x2)
        x2 = MaxPooling2D((2,2), strides=(2,2))(x2)
        conv_out2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x2)

        obj_inp3 = Input(batch_shape=(None, 32, 32, 1), name='restitution')
        x3 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(obj_inp3)
        # x3 = ZeroPadding2D((1,1))(x3)
        # x3 = MaxPooling2D((2,2), strides=(2,2))(x3)
        x3 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(x3)
        x3 = MaxPooling2D((2,2), strides=(2,2))(x3)
        x3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x3)
        x3 = MaxPooling2D((2,2), strides=(2,2))(x3)
        x3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x3)
        x3 = MaxPooling2D((2,2), strides=(2,2))(x3)
        conv_out3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x3)

        obj_inp4 = Input(batch_shape=(None, 32, 32, 1), name='vel_x')
        x4 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(obj_inp4)
        # x4 = ZeroPadding2D((1,1))(x4)
        # x4 = MaxPooling2D((2,2), strides=(2,2))(x4)
        x4 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(x4)
        x4 = MaxPooling2D((2,2), strides=(2,2))(x4)
        x4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x4)
        x4 = MaxPooling2D((2,2), strides=(2,2))(x4)
        x4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x4)
        x4 = MaxPooling2D((2,2), strides=(2,2))(x4)
        conv_out4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x4)

        obj_inp5 = Input(batch_shape=(None, 32, 32, 1), name='vel_y')
        x5 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(obj_inp5)
        # x5 = ZeroPadding2D((1,1))(x5)
        # x5 = MaxPooling2D((2,2), strides=(2,2))(x5)
        x5 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(x5)
        x5 = MaxPooling2D((2,2), strides=(2,2))(x5)
        x5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x5)
        x5 = MaxPooling2D((2,2), strides=(2,2))(x5)
        x5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x5)
        x5 = MaxPooling2D((2,2), strides=(2,2))(x5)
        conv_out5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x5)
        # v_merged = merge([x4, x5], mode='concat', concat_axis=1, name='vel_joint')
        # xv = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(v_merged)
        # xv = MaxPooling2D((2,2), strides=(2,2))(xv)
        # xv_out = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(xv)


        obj_inp6 = Input(batch_shape=(None, 32, 32, 1), name='acc_x')
        x6 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(obj_inp6)
        # x6 = ZeroPadding2D((1,1))(x6)
        # x6 = MaxPooling2D((2,2), strides=(2,2))(x6)
        x6 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(x6)
        x6 = MaxPooling2D((2,2), strides=(2,2))(x6)
        x6 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x6)
        x6 = MaxPooling2D((2,2), strides=(2,2))(x6)
        #x6 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x6)
        #x6 = MaxPooling2D((2,2), strides=(2,2))(x6)
        #conv_out6 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x6)

        obj_inp7 = Input(batch_shape=(None, 32, 32, 1), name='acc_y')
        x7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(obj_inp7)
        # x7 = ZeroPadding2D((1,1))(x7)
        # x7 = MaxPooling2D((2,2), strides=(2,2))(x7)
        x7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(x7)
        x7 = MaxPooling2D((2,2), strides=(2,2))(x7)
        x7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x7)
        x7 = MaxPooling2D((2,2), strides=(2,2))(x7)
        #x7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x7)
        #x7 = MaxPooling2D((2,2), strides=(2,2))(x7)
        #conv_out7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x7)
        a_merged = merge([x6, x7], mode='concat', concat_axis=1, name='acc_joint')
        xa = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(a_merged)
        xa = MaxPooling2D((2,2), strides=(2,2))(xa)
        xa_out = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(xa)

        # Merge is for layers, merge is for tensors
        concatenated = merge([conv_out0, conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, xa_out], mode='concat', concat_axis=1)

        o = Flatten()(concatenated)
        o1 = Dense(256*4, activation='relu', init=my_init)(o)
        o1 = Dense(256*2, activation='relu', init=my_init)(o1)
        out1 = Dense(1, activation='linear', init=my_init, name='dx')(o1)

        o2 = Dense(256*4, activation='relu', init=my_init)(o)
        o2 = Dense(256*2, activation='relu', init=my_init)(o2)
        out2 = Dense(1, activation='linear', init=my_init, name='dy')(o2)

    with tf.device('/cpu:0'):
        model = Model([obj_inp0, obj_inp1, obj_inp2, obj_inp3,
                       obj_inp4, obj_inp5, obj_inp6, obj_inp7], [out1, out2])


    if weights_path:
        with tf.device('/cpu:0'):
            model.load_weights(weights_path)
            print('Loaded weights from ' + weights_path)

    return model


 # or pass the h5 file for storing the model
inps = create_model('../src/models/inps2full_vxy.h5') # load the previously trained model
# inps.summary()
from keras.utils.visualize_util import plot
plot(inps, to_file='inps2_vxy.png', show_shapes=True)


def conv_input_batch_to_model(data):
    return [np.expand_dims(data[:,:,:,0], axis=3),
            np.expand_dims(data[:,:,:,1], axis=3),
            np.expand_dims(data[:,:,:,2], axis=3),
            np.expand_dims(data[:,:,:,3], axis=3),
            np.expand_dims(data[:,:,:,4], axis=3),
            np.expand_dims(data[:,:,:,5], axis=3),
            np.expand_dims(data[:,:,:,6], axis=3),
            np.expand_dims(data[:,:,:,7], axis=3)]

# def conv_output_batch_to_model(data):
#     return [data[:,0],
#             data[:,1]]


# x_unrolled_train = conv_input_batch_to_model(x_train)
# y_unrolled_train = conv_output_batch_to_model(y_train)


# Convert elements back to the original space
def to_real_data(data, log=False):
    # print('Data shape (before): ' + str(data.shape))
    # print('shape: ' + str(max_data.shape))
    data = np.multiply(data, max_data.reshape(2, 1, 1))
    data = data + min_data2.reshape((2, 1, 1))
    if (log):
        data = np.exp(data)
        data = data + min_data1.reshape((2, 1, 1))
    # print('Data shape (after): ' + str(data.shape))
    return data

def get_files():
    return glob.glob("../build/datatest/data*.npy")

old_files = []
files = get_files()
print('old_files ' + str(old_files))
print('files ' + str(files))

while True:
    files = get_files()
    if (set(old_files) == set(files)):
        print('Waiting for new files to be generated')
        t.sleep(0.01) # 10 ms
        continue
    # Now we have new files
    new_files = set(files) - set(old_files)
    print('New files: ' + str(new_files))
    # Read files and remove the mean
    bigdata = []
    for f in new_files:
        data = np.asarray(np.load(f))
        bigdata.append(data)

    bigdata = np.asarray(bigdata)
    print('np.asarray(bigdata)' + str(bigdata.shape))
    input = conv_input_batch_to_model(bigdata - mean_data)
    pred = inps.predict(input, batch_size=batch_size, verbose=1)
    pred_real = to_real_data(np.asarray(pred))
    print('Pred real: ' + str(pred_real.shape))
    i = 0
    for f in new_files:
        # print(filter(None, re.split('[_.]', f)))
        _, seed, id, _ = filter(None, re.split('[_.]', f))
        np.save('../build/datatest/pred_'+str(seed)+'_'+str(id)+'.npy',
            np.asarray([pred_real[0,i,:], pred_real[1,i,:]]))
        i += 1

    old_files = files
    print('Done')


# elem = 1
# print("Evaluating " + str(elem))
# pred = inps.predict(conv_input_batch_to_model(x_train[:elem]), batch_size=batch_size, verbose=1)

# pred = np.asarray(pred)
# pred_real = to_real_data(np.asarray(pred), log=False)
# y_unrolled_train = np.asarray(y_unrolled_train)


# Check accuracy of to from conversion!
# q1 = to_real_acc(np.asarray(y_train))
# q2 = np.asarray(unaltered_output)
# print(q2.shape)
# print(q1)
# print(q2)
# print(np.sum((q1 - q2)))


# print(pred.shape)
# print(pred_real.shape)
# print(pred_real[0,:elem,:])
# print(y_unrolled_train[0,:elem,:])
# print(y_unrolled_train.shape)
# print(y_unrolled_train)


# print(pred_real[0,:elem,0])
# print(q2[:elem,0])
# print(pred_real[0,:elem,0] - q2[:elem,0])

# print('---------------------------------------')
# print('   Prediction \t Training \t Diff (y\'-y)')
# print(np.concatenate((pred[0,:,:], y_unrolled_train[0,:elem,:], (pred[0,:,:] - y_unrolled_train[0,:elem,:])), axis=1))
# # print(np.concatenate((pred_real[0,:], y_unrolled_train[0,:elem], (pred_real[0,:]- y_unrolled_train[0,:elem])), axis=1))
# print('---------------------------------------')
# print(' real numbers: ')
# table = np.concatenate(([pred_real[0,:elem,0]], [q2[:elem,0]], [(pred_real[0,:elem,0] - q2[:elem,0])]), axis=0)
# # table = np.concatenate((table, [pred_real[1,:elem,0]], [q2[:elem,1]], [(pred_real[1,:elem,0] - q2[:elem,1])]), axis=0)
# print(table.transpose())
# print('---------------------------------------')
# print('---------------------------------------')
# print('---------------------------------------')
# print('Prediction \t Training \t Diff (y2\'-y2)')
# print(np.concatenate((pred[1,:,:], y_unrolled_train[1,:elem,:], (pred[1,:,:] - y_unrolled_train[1,:elem,:])), axis=1))
# print(' real numbers: ')
# table = np.concatenate(([pred_real[1,:elem,0]], [q2[:elem,1]], [(pred_real[1,:elem,0] - q2[:elem,1])]), axis=0)
# print(table.transpose())


# fix error on garbage collection race condition
from keras import backend as K
K.clear_session()
# import gc; gc.collect()
