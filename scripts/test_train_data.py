import numpy as np
from numpy.linalg import inv

import sim_world as sw
import model_creation as mc


inps, m2 = mc.create_model('/data/neuro_phys_sim/data/model.h5')

# Load data
X, Y = sw.getDataFromArchive('/data/neuro_phys_sim/data/data.hdf5', sample_from_data=True)
Xc = mc.conv_input_batch_to_model(X)
Yc = mc.conv_output_batch_to_model(Y)


Yc_pred = m2.predict(Xc)

Yc = np.transpose(np.asarray(Yc))
Yc_pred = np.transpose(np.asarray(Yc_pred))[0]

print Yc.shape
print Yc_pred.shape

counter = 0
for y, y_pred in zip(Yc, Yc_pred):
    # Get probability
    v = [y_pred[1], y_pred[3]]
    mu = [y_pred[0], y_pred[2]]
    prob = np.abs(np.matmul(inv(np.diag(v)), np.asarray(y - mu)))

    # print 'Prob: ', prob
    if (np.max(y) > 3):
        print 'FAR TOOO OUT !'
        counter += 1

print 'Counter: ', counter
print('Processed everything!')
