from keras.models import Model, Sequential
from keras.layers import Input, Lambda
from keras.layers.merge import concatenate
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import regularizers
from keras import backend as K

from keras.regularizers import l2

import tensorflow as tf
import numpy as np

# Remove info warning level msgs from tf
tf.logging.set_verbosity(tf.logging.ERROR) # for older versions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1' # speed up convolutions

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
            # x0 = Dropout(0.2)(x0)
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

        lam = 1e-6 # was 3 below
        o1 = Dense(256*4, activation='relu')(o) # kernel_initializer='random_uniform'
        o1 = Dense(256*2, activation='relu')(o1) # kernel_initializer='random_uniform'
        out1_mean = Dense(3, activation='linear', name='dx_mean', kernel_regularizer=regularizers.l2(lam))(o1) # W_regularizer=l2(lam)
        out1_var = Dense(1, activation='linear', name='dx_var')(o1)
        out1_alpha = Dense(1, activation='softmax', name='dx_alpha')(o1)
        out1 = Lambda(sampling, output_shape=(output_size,), name='dx')([out1_mean, out1_var])


        o2 = Dense(256*4, activation='relu')(o)
        o2 = Dense(256*2, activation='relu')(o2)
        out2_mean = Dense(1, activation='linear', name='dy_mean')(o2)
        out2_var = Dense(1, activation='linear', name='dy_var')(o2)
        out2 = Lambda(sampling, output_shape=(output_size,), name='dy')([out2_mean, out2_var])

    with tf.device(device_instance):
        model = Model(inpts, [out1, out2])
        # model = Model(inpts, [out1_mean, out2_mean])
        # model = Model(inpts, out1_mean)
        # m2 = Model(inpts, [out1_mean, out1_var, out2_mean, out2_var])
        # m2 = Model(inpts, out1_mean)
        m2 = Model(inpts, out1_mean)


    if weights_path:
        with tf.device(device_instance):
            model.load_weights(weights_path)
            print('Loaded weights from ' + weights_path)

    return model, m2

def plot_model(model):
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='/data/neuro_phys_sim/data/m2.png', show_shapes=True, show_layer_names=False)
    print 'Saved model image at /data/neuro_phys_sim/data/m2.png'

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

def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max),
                       axis=axis, keepdims=True))+x_max

def mean_log_Gaussian_like(y_true, parameters):
    """Mean Log Gaussian Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    'c' - number of outputs
    'm' - number of mixtures
    """
    c = 1
    m = 1
    components = K.reshape(parameters,[-1, c + 2, m])
    y_true = K.reshape(y_true,[-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = K.log(1 + K.exp(components[:, c, :]))
    K.clip(sigma, 1e-6, 1e3)
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha,1e-6,1.))

    exponent = K.log(alpha) - .5 * float(c) * K.log(2 * np.pi) \
            - float(c) * K.log(sigma) \
            - K.sum((y_true[:,:c,:] - mu)**2, axis=1)/(2*(sigma)**2)

            #     exponent = K.log(alpha) - .5 * float(c) * K.log(2 * np.pi) \
            # - float(c) * K.log(sigma) \
            # - K.sum((K.expand_dims(y_true,2) - mu)**2, axis=1)/(2*(sigma)**2)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res

def likelihood_loss(y_true, y_pred):
    # a, b = y_true
    # print a, b
    # print 'testtttt', y_true.shape, y_pred.shape, y_pred[:,0].shape, y_pred[:,1].shape, y_pred[2].shape
    # print 'Ytrue: ', y_true.get_shape(), K.shape(y_true)
    # out_size = y_true.get_shape().as_list()
    # print 'Some other method ', out_size, y_pred.get_shape().as_list()
    # # print 'True mean shape: ', K.shape(y_true), K.shape(y_true[:,0]), K.shape(y_true[:,1])
    mu_true = y_true[:,0]
    # dummy_zero = y_true[:,0]
    mu = y_pred[:,0]
    # sigmas = K.exp(y_pred[:,1])
    sigmas = K.log( 1 + K.exp(y_pred[:,1]))
    K.clip(sigmas, 1e-6, 1e3)
    # sigmas[sigmas < 1e-5]=1e-5 # clipping

    # print 'K.shape(y_pred) - ', K.shape(y_pred)
    # print '------'
    # print(YT1, YT2, YP1, YP2)

    # sample_prob

    # print (K.mean(K.square(y_pred - mu_true), axis=-1))

    # return K.mean(K.square(YP1 - YT1), axis=-1) +
    #        K.mean(K.square(YP2 - YT2), axis=-1)
    # o = -0.5 * K.log(K.square(sigmas)) - K.square(mu_true - mu) / (2.0 * K.square(sigmas))
    # print 'output: ', K.shape(o), K.ndim(o), K.ndim(K.mean(K.square(y_pred - mu_true), axis=-1))
    # return K.mean(K.square(y_pred - mu_true), axis=-1)

    s = K.exp(K.square(mu_true - mu)/(2*K.square(sigmas))) / (2 * np.pi * K.square(sigmas))
    s = K.maximum(s, 1e-20)
    loss =  -K.log(s)
    return loss
    # print K.shape(loss)
    # return K.maximum(loss, 1e-20)

    # return -0.5 * K.log(sigmas**2) - K.square(mu - mu_true) / (2.0 * sigmas**2)

    # z = ((mu_true - mu) / sigmas) ** 2 / -2.0
    # normalizer = sigmas * 2 * np.pi
    # z += - K.log(normalizer)
    # # return K.maximum(z, 1e-20)
    # return z


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
