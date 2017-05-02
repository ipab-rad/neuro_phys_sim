# import cv2
import numpy as np

from keras.models import Model

import sim_world as sw
import model_creation as mc

from sample import create_urandom_sample, draw_histograms, mhmc

inps, m2 = mc.create_model('data/model.h5')
# input_mean = np.load('data/mean_input.npy')

# min_outdata = np.load('data/min_outdata.npy')
# max_outdata = np.load('data/max_outdata.npy')

def data2model(data):
    # model_out =  mc.pred_model(inps, data)
    model_out =  mc.pred_model(m2, data)

    return [model_out[0], model_out[2]], [model_out[1], model_out[3]] # get the means values

x_all = np.empty(shape=(0, 2))
x_pred_all = np.empty(shape=(0, 2))

for i in xrange(1):
    samples, _ = mhmc(1, 1)
    x, x_pred, crops, outs = sw.simulateWithModel(samples[-1], data2model)
    print "Add new values len: ", len(crops)
    x_all = np.concatenate((x_all, np.asarray(x)))
    x_pred_all = np.concatenate((x_pred_all, np.asarray(x_pred)))


print 'All delta x size: ', x_all.shape

draw_histograms(x_all[:,0], "eval_x")
draw_histograms(x_pred_all[:,0], "eval_x_pred")

print('Processed everything!')
