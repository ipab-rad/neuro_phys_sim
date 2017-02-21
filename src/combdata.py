# import cv2
import numpy as np
# from keras.models import Model, Sequential
# from keras.layers import Input, Lambda, merge
# from keras.layers.core import Flatten, Dense, Dropout
# from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
# from keras.optimizers import SGD, Adam, Adamax

# # MPL no display backend
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# # end

# FIX
# import tensorflow as tf
# tf.python.control_flow_ops = tf
# END fix

import re
import glob

model_dir = '/src/inps/src/models/data2'
core_data_dir1 = '/src/inps/build/data'
core_data_dir2 = '/src/inps/build/data2'
core_data_dir3 = '/src/inps/build/datacol'

files = glob.glob(core_data_dir1 + '/data*.npy') + glob.glob(core_data_dir2 + '/data*.npy') + glob.glob(core_data_dir3 + '/data*.npy')

files = sorted(files)
print('Sorted data.')

bigdata = []
bigresp = []

# thr = 9.81*5 # max g = 5g

for f in files[:]:
    ldir, seed, id, _ = filter(None, re.split('[_.]', f))
    data = np.asarray(np.load(f))
    resp = np.asarray(np.load(ldir[:-4] + 'resp_' +
                      str(seed) + '_' + str(id) + '.npy'))
    subsetresp = resp[5:7]
    # if (max(abs(subsetresp)) > thr):
    #     print('Tresholding to ' + str(thr) + '!')
    #     subsetresp = np.asarray(subsetresp)
    #     subsetresp[subsetresp > thr]=thr
    #     subsetresp[subsetresp < -thr]=-thr
    #     # continue
    bigdata.append(data)
    bigresp.append(subsetresp)
    # Give feedbakc on loading
    if (len(bigdata) % 5000 == 0):
        print('Loaded ' + str(len(bigdata)) + ' samples.')

bigdata = np.asarray(bigdata)
bigresp = np.asarray(bigresp)
# unaltered_output = bigresp

# log the data
# min_data = np.min(bigresp.flatten())
# min_data = np.min(bigresp, axis = 0)
# print('Min data: ' + str(min_data))

# bigresp = bigresp + (np.abs(min_data)*1.1)
# bigresp = np.log(bigresp)


# find mean and shift the data
resp_mean = np.mean(bigresp, axis = 0)
# resp_mean = np.mean(bigresp)
print('Mean vector: ' + str(resp_mean))
bigresp = (bigresp - resp_mean)*100


print('Min data (before): ' + str(np.min(bigresp, axis = 0)))
print('Max data (before): ' + str(np.max(bigresp, axis = 0)))

# # normalize data to 0-1
# min_data1 = np.minimum(np.min(bigresp, axis = 0)*1.1, 0.1)
# bigresp = bigresp - min_data1
# print('Shifted lower bound by ' + str(min_data1))
# # data is already in range [0.1 - N)
# bigresp = np.log(bigresp) # take the log
# print('Min data (after log): ' + str(np.min(bigresp, axis = 0)))
# print('Max data (after log): ' + str(np.max(bigresp, axis = 0)))

min_data2 = np.min(bigresp, axis = 0)
np.save(model_dir + '/min_data2full.npy', min_data2)
bigresp = bigresp - min_data2
# now data is in range [0, ...)
print('Shifted lower bound by (after log): ' + str(min_data2))

max_data = np.max(bigresp, axis=0)
np.save(model_dir + '/max_data2full.npy', max_data)
print('Max data to norm dist ' + str(max_data))
bigresp = bigresp / max_data
# now data is in range [0 .. 1]

print('Min data:  ' + str(np.min(bigresp, axis = 0)))
print('Max data:  ' + str(np.max(bigresp, axis = 0)))
print('Mean data: ' + str(np.mean(bigresp, axis = 0)))

# Shift input mean data
# print('Mean input data: ' + str(np.mean(bigdata, axis = 0).shape))
inp_data_mean = np.mean(bigdata, axis = 0)
bigdata = bigdata - inp_data_mean
np.save(model_dir + '/mean_input_data2full.npy', inp_data_mean)
print('Mean Data shape: '  + str(inp_data_mean.flatten().shape))
print('Input data shifted')

print('Data loaded.')

print('Data: ' + str(bigdata.shape))
print('Resp: ' + str(bigresp.shape))

# import matplotlib.pyplot as plt
# for x in range(0,np.size(bigresp, 1)):
#     # print(bigresp[:,x].shape)
#     print('Max data: ' + str(np.max(bigresp[:,x])))
#     print('Min data: ' + str(np.min(bigresp[:,x])))
#     plt.hist(bigresp[:,x], bins=20)  # plt.hist passes it's arguments to np.histogram
#     plt.title("Histogram with 20 bins")
#     filename = "hist_" + str(x) + ".png"
#     plt.savefig(filename)


print('Shuffling data') # maybe shuffle which order to read the files?
from random import shuffle, seed
index_shuf = list(range(len(bigdata)))
seed(0)
shuffle(index_shuf)
bigdata = [bigdata[i] for i in index_shuf]
bigresp = [bigresp[i] for i in index_shuf]
# unaltered_output = [unaltered_output[i] for i in index_shuf]
print('Done shuffling.')


# Using saved_data!
x_train = np.asarray(bigdata)
y_train = np.asarray(bigresp)
# x_unrolled_train = conv_input_batch_to_model(x_train)
# y_unrolled_train = conv_output_batch_to_model(y_train)

# ################
# # Load data
# modeldir = '../src/models/'
# # bigdata = []
# # bigresp = []

# # Get params
# min_data2 = np.load(modeldir + '/min_data2full.npy') #np.asarray([-6.64888525, -13.71338463])
# max_data = np.load(modeldir + '/max_data2full.npy') #np.asarray([12.40662193, 19.73525047])
# mean_data = np.load(modeldir + '/mean_input_data2full.npy')
# mean_data = mean_data.reshape((1, 32, 32, 8))
# print('Min data:  ' + str(min_data2))
# print('Max data:  ' + str(max_data))
# print('Mean data:  ' + str(mean_data.shape))

# x_unrolled_train = np.load(modeldir + '/x_unrolled_train.npy')
# x_unrolled_train = [x_unrolled_train[0,:], x_unrolled_train[1,:], x_unrolled_train[2,:],x_unrolled_train[3,:],x_unrolled_train[4,:],x_unrolled_train[5,:],x_unrolled_train[6,:],x_unrolled_train[7,:]]
# y_unrolled_train = np.load(modeldir + '/y_unrolled_train.npy')
# y_unrolled_train = [y_unrolled_train[0,:], y_unrolled_train[1,:]]
################

# print(len(x_unrolled_train))
# print(x_unrolled_train[0].shape)
print(x_train.shape)
print(y_train.shape)

# Save data to a single file
print('Saving...')
np.save(model_dir + '/x_train.npy', x_train)
np.save(model_dir + '/y_train_v.npy', y_train)
print('SAVED ALL THE DATA. ARE YOU SURE YOU NEED TO DO IT EVERY TIME? COMMENT ME OUT.')

# print(pred_real.shape)
# print(pred_real[0,:elem,:])
# print(y_unrolled_train[0,:elem,:])
# print(y_unrolled_train)

# fix error on garbage collection race condition
# from keras import backend as K
# K.clear_session()
# import gc; gc.collect()
