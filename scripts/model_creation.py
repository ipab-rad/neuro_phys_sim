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

        lam = 1e-6
        o1 = Dense(256*4, activation='relu')(o) # kernel_initializer='random_uniform'
        o1 = Dense(256*2, activation='relu')(o1) # kernel_initializer='random_uniform'
        out1_mean = Dense(3, activation='linear', name='dx', kernel_regularizer=regularizers.l2(lam))(o1) # W_regularizer=l2(lam)
        # out1_var = Dense(1, activation='linear', name='dx_var')(o1)
        # out1_alpha = Dense(1, activation='softmax', name='dx_alpha')(o1)
        # out1 = Lambda(sampling, output_shape=(output_size,), name='dxo')([out1_mean, out1_var])


        o2 = Dense(256*4, activation='relu')(o)
        o2 = Dense(256*2, activation='relu')(o2)
        out2_mean = Dense(3, activation='linear', name='dy', kernel_regularizer=regularizers.l2(lam))(o2)
        # out2_var = Dense(1, activation='linear', name='dy_var')(o2)
        # out2 = Lambda(sampling, output_shape=(output_size,), name='dyo')([out2_mean, out2_var])

    with tf.device(device_instance):
        model = Model(inpts, [out1_mean, out2_mean])
        # model = Model(inpts, [out1_mean, out2_mean])
        # m2 = Model(inpts, out1_mean)

    if weights_path:
        with tf.device(device_instance):
            model.load_weights(weights_path)
            print('Loaded weights from ' + weights_path)

    return model

def plot_model(model):
    filename = '/data/neuro_phys_sim/data/m2.png'
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=False)
    print 'Saved model image at ', filename

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

def conv_output_batch_to_model(data, parts=2):
    res = []
    for i in xrange(parts): # add extra zero arrays for var and alpha
        oval = np.asarray(data[:,i]).reshape(-1, 1)
        Yi = np.zeros(shape=(oval.shape[0], oval.shape[1]*3), dtype=np.float32)
        Yi[:oval.shape[0], :oval.shape[1]] = oval
        res.append(Yi)
    return res

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

    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res
