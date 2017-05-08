from keras.models import Model, Sequential
from keras.layers import Input, Lambda
from keras.layers.merge import concatenate
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K

import tensorflow as tf
import numpy as np

# Remove info warning level msgs from tf
tf.logging.set_verbosity(tf.logging.ERROR) # for older versions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# device_instance = '/cpu:0'
device_instance = '/gpu:1'

# Change tf params
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# end params change

output_size = 1

def sampling(args):
    epsilon_std = 1.0
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], output_size),
                              mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def create_model(weights_path=None):
    with tf.device(device_instance):
        outs = []
        inpts = []

        for i in xrange(7):
            obj_inp = Input(batch_shape=(None, 32, 32, 1))
            inpts.append(obj_inp)
            x0 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(obj_inp)
            x0 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(x0)
            x0 = Dropout(0.2)(x0)
            x0 = MaxPooling2D((2,2), strides=(2,2))(x0)
            x0 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x0)
            x0 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x0)
            x0 = MaxPooling2D((2,2), strides=(2,2))(x0)
            x0 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x0)
            x0 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x0)
            x0 = MaxPooling2D((2,2), strides=(2,2))(x0)
            outs.append(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x0))

        # Merge is for layers, merge is for tensors
        concatenated = concatenate(outs, axis=1)
        o = Flatten()(concatenated)

        o1 = Dense(256*4, activation='relu')(o)
        o1 = Dense(256*2, activation='relu')(o1)
        out1_mean = Dense(1, activation='linear', name='dx_mean')(o1)
        out1_var = Dense(1, activation='linear', name='dx_var')(o1)
        out1 = Lambda(sampling, output_shape=(output_size,), name='dx')([out1_mean, out1_var])


        o2 = Dense(256*4, activation='relu')(o)
        o2 = Dense(256*2, activation='relu')(o2)
        out2_mean = Dense(1, activation='linear', name='dy_mean')(o2)
        out2_var = Dense(1, activation='linear', name='dy_var')(o2)
        out2 = Lambda(sampling, output_shape=(output_size,), name='dy')([out2_mean, out2_var])

    with tf.device(device_instance):
        model = Model(inpts, [out1, out2])
        # model = Model(inpts, [out1_mean, out2_mean])
        # model = Model(inpts, out1)
        m2 = Model(inpts, [out1_mean, out1_var, out2_mean, out2_var])


    if weights_path:
        with tf.device(device_instance):
            model.load_weights(weights_path)
            print('Loaded weights from ' + weights_path)

    return model, m2

def plot_model(model):
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='inps2full.png', show_shapes=True, show_layer_names=False)

def pred_model(model, data):
    return model.predict(data)

def conv_input_batch_to_model(data):
    return [np.expand_dims(data[:,0,:,:], axis=3),
            np.expand_dims(data[:,1,:,:], axis=3),
            np.expand_dims(data[:,2,:,:], axis=3),
            np.expand_dims(data[:,3,:,:], axis=3),
            np.expand_dims(data[:,4,:,:], axis=3),
            np.expand_dims(data[:,5,:,:], axis=3),
            np.expand_dims(data[:,6,:,:], axis=3)]

def conv_output_batch_to_model(data):
    return [data[:,0],
            data[:,1]] # for dx dy
    # return data[:,0]
    # return [data[:,3],
    #         data[:,4]] # for dx dy

def likelihood_loss(y_true, y_pred):
    print(y_true.get_shape())
    out_size = y_true.get_shape().as_list()[1]
    # YT1 = y_true[:,0]
    # YT2 = y_true[:,1]
    # YP1 = y_pred[:,0]
    # YP2 = y_pred[:,1]

    print K.shape(y_pred)
    print '------'
    # print(YT1, YT2, YP1, YP2)

    # sample_prob

    # print (K.mean(K.square(y_pred - y_true), axis=-1))

    # return K.mean(K.square(YP1 - YT1), axis=-1) +
    #        K.mean(K.square(YP2 - YT2), axis=-1)

    return K.mean(K.square(y_pred - y_true), axis=-1)

def preprocess_data(data):
    print 'data.shape: ', data.shape
    data_mean = np.mean(data, axis = 0)
    data = data - data_mean
    np.save('data/mean_input.npy', data_mean)
    print('Mean Data shape: '  + str(data_mean.shape))
    # print data_mean[0]
    # print data_mean[1]


    import cv2
    for i in range(7):
        cv2.imwrite("verify"+str(i)+".png", data[0][i]*255)

    for i in range(32):
        print data[0][0][i]

    print '------------------------------------------------------------'
    for i in range(32):
        print data_mean[0][i]

    print('Input data shifted')

    print (np.mean(data))

    # from sklearn import preprocessing
    # print data[0]
    # print data[:,0].shape
    # data = [preprocessing.scale(data[:,i]) for i in xrange(len(data))]
    # print data
    return data

def preprocess_output(o):
    print '-----'
    print len(o)
    print o[0].shape
    print 'Values larger than 10: ', sum(i[0] > 10 for i in o)
    print 'Values larger than 10: ', sum(i[1] > 10 for i in o)
    print 'max: ', np.max(o)
    print 'min: ', np.min(o)
    print '-----'

    for i in o:
        if i[0] > 1.0:
            i[0] = 1.0
        elif i[0] < -1.0:
            i[0] = -1.0

        if i[1] > 1.0:
            i[1] = 1.0
        elif i[1] < -1.0:
            i[1] = -1.0

    print 'Values larger than 1: ', sum(i[0] >= 1 for i in o)
    print 'Values larger than 1: ', sum(i[1] >= 1 for i in o)
    print 'max: ', np.max(o)
    print 'min: ', np.min(o)
    print '-----'

    # # from sklearn.preprocessing import MinMaxScaler
    # # o = MinMaxScaler().fit_transform(o)

    # min_data = np.min(o, axis = 0)
    # np.save('data/min_outdata.npy', min_data)
    # o = o - min_data
    # # now data is in range [0, ...)
    # # print('Shifted lower bound by (after log): ' + str(min_data))

    # max_data = np.max(o, axis=0)
    # np.save('data/max_outdata.npy', max_data)
    # print('Max data to norm dist ' + str(max_data))
    # o = o / max_data
    return o
