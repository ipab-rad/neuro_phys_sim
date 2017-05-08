import numpy as np
from numpy.linalg import inv

import sim_world as sw
import model_creation as mc

inps, m2 = mc.create_model('/data/neuro_phys_sim/data/model.h5')
# inps, m2 = mc.create_model('/data/neuro_phys_sim/data/model_refited_10.h5')

# Load data
X, Y = sw.getDataFromArchive('/data/neuro_phys_sim/data/data_300.hdf5', sample_from_data=False)
# X, Y = sw.getDataFromArchive('/data/neuro_phys_sim/data/extra_data.hdf5', sample_from_data=False)
print 'Data loaded.'
Xc = mc.conv_input_batch_to_model(X)
Yc = mc.conv_output_batch_to_model(Y)
print 'Data converted.'

# # Loss and metrics
# inps.compile(optimizer='adam', loss='mse')
# loss_and_metrics = inps.evaluate(Xc, Yc, batch_size=8*256, verbose=1)
# print('Total loss: ' + str(loss_and_metrics))

# exit(0)
Yc_pred = m2.predict(Xc, batch_size=8*256)
print 'Network prediction done.'

Yc = np.transpose(np.asarray(Yc))
Yc_pred = np.transpose(np.asarray(Yc_pred))[0]

print Yc.shape
print Yc_pred.shape

print 'Evaluating performance...'
counter = 0
for y, y_pred in zip(Yc, Yc_pred):
    # Get probability
    mu = [y_pred[0], y_pred[2]]
    s = [y_pred[1], y_pred[3]]
    var = np.exp(np.asarray(s, dtype='float64').flatten()/2.0)
    prob = np.abs(np.matmul(inv(np.diag(var)), np.asarray(y - mu)))

    # print y, mu, var
    # print 'Prob: ', prob
    if (np.max(prob) > 3):
        # print 'Far too unprobable - ', prob
        counter += 1

print 'Above threshold occurances: ', counter, ' / ', len(Yc), \
      ' (', float(counter) / len(Yc) , ')'
print('Processing done!')
