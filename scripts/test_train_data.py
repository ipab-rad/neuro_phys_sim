import numpy as np
from numpy.linalg import inv

import sim_world as sw
import model_creation as mc

m2 = mc.create_model('/data/neuro_phys_sim/data/modelu2.h5')
# inps, m2 = mc.create_model('/data/neuro_phys_sim/data/model_refited_10.h5')

# Load data
X, Y = sw.getDataFromArchive('/data/neuro_phys_sim/data/data_300.hdf5', sample_from_data=False)
# X, Y = sw.getDataFromArchive('/data/neuro_phys_sim/data/extra_data.hdf5', sample_from_data=False)
print 'Data loaded.'
Xc = mc.conv_input_batch_to_model(X)
Yc = mc.conv_output_batch_to_model(Y)
print 'Data converted. Predicting ...'

# # Loss and metrics
# inps.compile(optimizer='adam', loss='mse')
# loss_and_metrics = inps.evaluate(Xc, Yc, batch_size=8*256, verbose=1)
# print('Total loss: ' + str(loss_and_metrics))

# exit(0)
Yc_pred = m2.predict(Xc, batch_size=8*256, verbose=1)
print 'Network prediction done.'

# print Yc_pred, Yc

Yc = np.transpose(np.asarray(Yc))
Yc_pred = np.transpose(np.asarray(Yc_pred))

print Yc.shape
print Yc_pred.shape

print 'Evaluating performance...'
counter = [0, 0, 0, 0, 0]
sum_prob = 0
error = []
for y, mu, s, a in zip(Yc[0], Yc_pred[0], Yc_pred[1], Yc_pred[2]):
    # print y_pred
    # print y
    # Get probability
    # mu = [y_pred[0], y_pred[2]]
    # s = [y_pred[1], y_pred[3]]
    # var = np.exp(np.asarray(s, dtype='float64').flatten()/2.0)
    # mu = [y_pred[0]]
    # s = [y_pred[1]]
    var = np.log(1 + np.exp(s))
    prob = np.abs(np.matmul(inv(np.diag(var)), np.asarray(y - mu)))
    error.append(np.asarray(y - mu))
    # print y, mu, var
    # print 'Prob: ', prob
    max_prob = np.max(prob)
    sum_prob += max_prob
    for i in xrange(len(counter)):
        if (max_prob > (i+1)):
            # print 'Far too unprobable - ', prob
            counter[i] += 1

print 'Above sigma threshold occurances \n[1s, 2s, 3s, ...]:\n', counter, ' / ', len(Yc[0]), \
      '\n', [float(c) / len(Yc[0]) for c in counter], '\n'
print 'Average sigma prob: ', sum_prob / len(Yc[0])
print 'MSE: ', np.square(np.asarray(error)).mean()
print('Processing done!')
