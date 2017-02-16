# import cv2
import numpy as np
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
files =  glob.glob("../build/data/data*.npy")

bigdata = []
bigresp = []

thr = 9.81*5 # max g = 5g

for f in files[:]:
    _, seed, id, _ = filter(None, re.split('[_.]', f))
    data = np.asarray(np.load(f))
    resp = np.asarray(np.load('../build/data/resp_'+str(seed)+'_'+str(id)+'.npy'))
    subsetresp = resp[3:5]
    if (max(abs(subsetresp)) > thr):
        print('Tresholding to ' + str(thr) + '!')
        subsetresp = np.asarray(subsetresp)
        subsetresp[subsetresp > thr]=thr
        subsetresp[subsetresp < -thr]=-thr
        # continue
    bigdata.append(data)
    bigresp.append(subsetresp)
    # Give feedbakc on loading
    if (len(bigdata) % 10000 == 0):
        print('Loaded ' + str(len(bigdata)) + ' samples.')

bigdata = np.asarray(bigdata)
bigresp = np.asarray(bigresp)
unaltered_output = bigresp

# log the data
# min_data = np.min(bigresp.flatten())
# min_data = np.min(bigresp, axis = 0)
# print('Min data: ' + str(min_data))

# bigresp = bigresp + (np.abs(min_data)*1.1)
# bigresp = np.log(bigresp)


# # find mean and shift the data
# resp_mean = np.mean(bigresp, axis = 0)
# # resp_mean = np.mean(bigresp)
# print('Mean vector: ' + str(resp_mean))
# bigresp = (bigresp - resp_mean)*100


print('Min data (before): ' + str(np.min(bigresp, axis = 0)))
print('Max data (before): ' + str(np.max(bigresp, axis = 0)))

# normalize data to 0-1
min_data1 = np.minimum(np.min(bigresp, axis = 0)*1.1, 0.1)
bigresp = bigresp - min_data1
print('Shifted lower bound by ' + str(min_data1))
# data is already in range [0.1 - N)
bigresp = np.log(bigresp) # take the log
print('Min data (after log): ' + str(np.min(bigresp, axis = 0)))
print('Max data (after log): ' + str(np.max(bigresp, axis = 0)))

min_data2 = np.min(bigresp, axis = 0)
bigresp = bigresp - min_data2
# now data is in range [0, ...)
print('Shifted lower bound by (after log): ' + str(min_data2))

max_data = np.max(bigresp, axis=0)
print('Max data to norm dist ' + str(max_data))
bigresp = bigresp / max_data
# now data is in range [0 .. 1]

print('Min data:  ' + str(np.min(bigresp, axis = 0)))
print('Max data:  ' + str(np.max(bigresp, axis = 0)))
print('Mean data: ' + str(np.mean(bigresp, axis = 0)))

# Shift input mean data
# print('Mean input data: ' + str(np.mean(bigdata, axis = 0)))
bigdata = bigdata - np.mean(bigdata, axis = 0)
np.mean(bigdata, axis = 0).tofile('mean_data2full.dat')
print('Input data shifted')

print('Data loaded.')

print('Data: ' + str(bigdata.shape))
print('Resp: ' + str(bigresp.shape))

import matplotlib.pyplot as plt
for x in range(0,np.size(bigresp, 1)):
    # print(bigresp[:,x].shape)
    print('Max data: ' + str(np.max(bigresp[:,x])))
    print('Min data: ' + str(np.min(bigresp[:,x])))
    plt.hist(bigresp[:,x], bins=20)  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram with 20 bins")
    filename = "hist_" + str(x) + ".png"
    plt.savefig(filename)


print('shuffling data') # maybe shuffle which order to read the files?
from random import shuffle, seed
index_shuf = list(range(len(bigdata)))
seed(0)
shuffle(index_shuf)
bigdata = [bigdata[i] for i in index_shuf]
bigresp = [bigresp[i] for i in index_shuf]
unaltered_output = [unaltered_output[i] for i in index_shuf]
print('Done shuffling.')


# Generate model
batch_size = 256*4
# latent_dim = len(classes)
nb_epoch = 50 #250*4
# latent_dim = 64

x_train = np.asarray(bigdata)
y_train = np.asarray(bigresp)

from keras import initializations
def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def create_model(weights_path=None):
    # with tf.device('/cpu:0'):
    # inp = Input(batch_shape=(None, 32, 32, 1))
    # x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', init=my_init)(inp)
    # x = Convolution2D(16, 8, 8, activation='relu', border_mode='same', init=my_init)(x)
    # x = Convolution2D(32, 4, 4, activation='relu', border_mode='same', init=my_init)(x)
    # # # model.add(Dropout(0.15))
    # # # x = Dropout(0.25)(x)
    # # # x = Dropout(0.25)(x)
    # last_submodel = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x)
    # # x = Flatten()(x)
    # # last_submodel = Dense(178, activation='relu', init=my_init)(x)

    # input_conv_model = Model(inp, last_submodel)
    # input_conv_model.summary()

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
    # x4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x4)
    # x4 = MaxPooling2D((2,2), strides=(2,2))(x4)
    # conv_out4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x4)

    obj_inp5 = Input(batch_shape=(None, 32, 32, 1), name='vel_y')
    x5 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(obj_inp5)
    # x5 = ZeroPadding2D((1,1))(x5)
    # x5 = MaxPooling2D((2,2), strides=(2,2))(x5)
    x5 = Convolution2D(16, 3, 3, activation='relu', border_mode='same', init=my_init)(x5)
    x5 = MaxPooling2D((2,2), strides=(2,2))(x5)
    x5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x5)
    x5 = MaxPooling2D((2,2), strides=(2,2))(x5)
    # x5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x5)
    # x5 = MaxPooling2D((2,2), strides=(2,2))(x5)
    # conv_out5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(x5)
    v_merged = merge([x4, x5], mode='concat', concat_axis=1, name='vel_joint')
    xv = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(v_merged)
    xv = MaxPooling2D((2,2), strides=(2,2))(xv)
    xv_out = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init=my_init)(xv)


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
    concatenated = merge([conv_out0, conv_out1, conv_out2, conv_out3, xv_out, xa_out], mode='concat', concat_axis=1)

    o = Flatten()(concatenated)
    o1 = Dense(256*4, activation='relu', init=my_init)(o)
    o1 = Dense(256*2, activation='relu', init=my_init)(o1)
    out1 = Dense(1, activation='linear', init=my_init, name='dx')(o1)

    o2 = Dense(256*4, activation='relu', init=my_init)(o)
    o2 = Dense(256*2, activation='relu', init=my_init)(o2)
    out2 = Dense(1, activation='linear', init=my_init, name='dy')(o2)

    #with tf.device('/cpu:0'):
    model = Model([obj_inp0, obj_inp1, obj_inp2, obj_inp3,
                   obj_inp4, obj_inp5, obj_inp6, obj_inp7], [out1, out2])


    if weights_path:
        model.load_weights(weights_path)
        print('Loaded weights from ' + weights_path)

    return model


 # or pass the h5 file for storing the model
inps = create_model('inps2full.h5') # load the previously trained model
inps.summary()
from keras.utils.visualize_util import plot
plot(inps, to_file='inps2merged.png', show_shapes=True)

# optimz = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
optimz = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
# optimz = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# optimz = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

inps.compile(optimizer=optimz, loss='mse') # , metrics=['accuracy']


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, verbose=1,
              patience=5, cooldown=10, min_lr=0.000001)

def conv_input_batch_to_model(data):
    return [np.expand_dims(data[:,:,:,0], axis=3),
            np.expand_dims(data[:,:,:,1], axis=3),
            np.expand_dims(data[:,:,:,2], axis=3),
            np.expand_dims(data[:,:,:,3], axis=3),
            np.expand_dims(data[:,:,:,4], axis=3),
            np.expand_dims(data[:,:,:,5], axis=3),
            np.expand_dims(data[:,:,:,6], axis=3),
            np.expand_dims(data[:,:,:,7], axis=3)]

def conv_output_batch_to_model(data):
    return [data[:,0],
            data[:,1]]


x_unrolled_train = conv_input_batch_to_model(x_train)
y_unrolled_train = conv_output_batch_to_model(y_train)

print(x_train.shape)
print(len(x_unrolled_train))
print(x_unrolled_train[0].shape)

samp_weight = (np.abs(np.mean(y_unrolled_train, axis=0) - np.abs(y_unrolled_train))*10000)*10 #np.ones((len(y_train), )) # sample different instances of the sim
# the last *10 only for acceleration
# samp_weight = np.ones(len(y_train))
# print(samp_weight)

# Full train
hist = inps.fit(x_unrolled_train, y_unrolled_train,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    shuffle=True,  # does not affect valdiation set
    validation_split=0.1,
#    sample_weight = samp_weight,
    callbacks=[reduce_lr])
    # callbacks=[early_stopping])

# Loss and metrics
loss_and_metrics = inps.evaluate(conv_input_batch_to_model(x_train), y_unrolled_train, batch_size=batch_size, verbose=1)
print('Total loss: ' + str(loss_and_metrics))

# Saving it later
# inps.save('inps.h5')  # creates a HDF5 file
# print('Saved model!')


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



elem = 50
print("First " + str(elem))
pred = inps.predict(conv_input_batch_to_model(x_train[:elem]), batch_size=batch_size, verbose=1)

pred = np.asarray(pred)
pred_real = to_real_data(np.asarray(pred), log=True)
y_unrolled_train = np.asarray(y_unrolled_train)


# Check accuracy of to from conversion!
# q1 = to_real_acc(np.asarray(y_train))
q2 = np.asarray(unaltered_output)
print(q2.shape)
# print(q1)
# print(q2)
# print(np.sum((q1 - q2)))


print(pred.shape)
# print(pred_real.shape)
# print(pred_real[0,:elem,:])
# print(y_unrolled_train[0,:elem,:])
print(y_unrolled_train.shape)
# print(y_unrolled_train)


# print(pred_real[0,:elem,0])
# print(q2[:elem,0])
# print(pred_real[0,:elem,0] - q2[:elem,0])

print('---------------------------------------')
print('   Prediction \t Training \t Diff (y\'-y)')
print(np.concatenate((pred[0,:,:], y_unrolled_train[0,:elem,:], (pred[0,:,:] - y_unrolled_train[0,:elem,:])), axis=1))
# print(np.concatenate((pred_real[0,:], y_unrolled_train[0,:elem], (pred_real[0,:]- y_unrolled_train[0,:elem])), axis=1))
print('---------------------------------------')
print(' real numbers: ')
table = np.concatenate(([pred_real[0,:elem,0]], [q2[:elem,0]], [(pred_real[0,:elem,0] - q2[:elem,0])]), axis=0)
# table = np.concatenate((table, [pred_real[1,:elem,0]], [q2[:elem,1]], [(pred_real[1,:elem,0] - q2[:elem,1])]), axis=0)
print(table.transpose())
print('---------------------------------------')
print('---------------------------------------')
print('---------------------------------------')
print('Prediction \t Training \t Diff (y2\'-y2)')
print(np.concatenate((pred[1,:,:], y_unrolled_train[1,:elem,:], (pred[1,:,:] - y_unrolled_train[1,:elem,:])), axis=1))
print(' real numbers: ')
table = np.concatenate(([pred_real[1,:elem,0]], [q2[:elem,1]], [(pred_real[1,:elem,0] - q2[:elem,1])]), axis=0)
print(table.transpose())


loss_and_metrics = inps.evaluate(conv_input_batch_to_model(x_train), conv_output_batch_to_model(y_train), batch_size=batch_size, verbose=1)
print('Total loss: ' + str(loss_and_metrics))

# Draw some more histograms
# Acceleration along x
plt.hist(pred_real[0,:,:], bins=20, range=[-5, 5], facecolor='red')  # plt.hist passes it's arguments to np.histogram
plt.savefig('acc_hist_0_only_predicted.png')
plt.hist(q2[:,0], bins=20, range=[-5, 5], facecolor='green')
plt.title("Acc_x histogram with 20 bins")
filename = "acc_hist_x.png"
plt.savefig(filename)
plt.clf()
# Acceleration along y
plt.hist(pred_real[1,:,:], bins=20, range=[-5, 5], facecolor='red')  # plt.hist passes it's arguments to np.histogram
plt.savefig('acc_hist_1_only_predicted.png')
plt.hist(q2[:,1], bins=20, range=[-5, 5], facecolor='green')
plt.title("Acc_y histogram with 20 bins")
filename = "acc_hist_y.png"
plt.savefig(filename)


inps.save('inps2full.h5')  # creates a HDF5 file
print('Saved model!')

# fix error on garbage collection race condition
from keras import backend as K
K.clear_session()
# import gc; gc.collect()
