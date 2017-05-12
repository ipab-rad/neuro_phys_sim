import numpy as np
import time as t
from keras.models import Model, Sequential
from keras.layers import Input, Lambda
from keras.layers.merge import concatenate
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, Adamax
from keras import backend as K

import sim_world as sw
import model_creation as mc
from sample import draw_histograms

# MPL no display backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# end

# Generate model
batch_size = 256*8 # 256
nb_epoch = 300

inps = mc.create_model() # load the previously trained model '/data/neuro_phys_sim/data/model.h5'
# inps.compile(optimizer='adam', loss='mse') # , metrics=['accuracy']
# inps.compile(optimizer=optimz, loss='mse') # , metrics=['accuracy']
# m2.compile(optimizer='adam', loss=mc.likelihood_loss) # , metrics=['accuracy']

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, verbose=1,
                              patience=5, cooldown=10, min_lr=0.000001)

# Plot model
# inps.summary()
mc.plot_model(inps)

# Load data
print 'Loading data ...'
X, Y = sw.getDataFromArchive('/data/neuro_phys_sim/data/data_300.hdf5',
                             sample_from_data=False)

Xc = mc.conv_input_batch_to_model(X)
Yc = mc.conv_output_batch_to_model(Y)

# draw_histograms(Yc[0][0], "x")
# draw_histograms(Yc[1][0], "y")

print Xc[0].shape
print Yc[0].shape

print 'Optimizing mean and variance with custom loss!'
optimizer = Adamax(clipnorm=1e-6, clipvalue=1e6)
inps.compile(optimizer=optimizer, loss=mc.mean_log_Gaussian_like)

# Full train
hist = inps.fit(Xc, Yc,
    batch_size=batch_size,
    epochs=nb_epoch,
    shuffle=True,  # does not affect valdiation set
    validation_split=0.1,
    callbacks=[reduce_lr]
    )

# Loss and metrics
inps.compile(optimizer='adam', loss='mse')
loss_and_metrics = inps.evaluate(Xc, Yc, batch_size=batch_size, verbose=1)
print('Total loss: ' + str(loss_and_metrics))

print 'Using model to predict ...'
Yc_pred = inps.predict(Xc, batch_size=8*256, verbose=2)

Yc = np.transpose(np.asarray(Yc))
Yc_pred = np.transpose(np.asarray(Yc_pred))

print Yc_pred
id = 0
for y, mu, s, a in zip(Yc[0], Yc_pred[0], Yc_pred[1], Yc_pred[2]):
    print '------'
    # print y_pred
    # Get probability
    # mu = [y_pred[0], y_pred[3]]
    # s = [y_pred[1], y_pred[4]]
    # a = [y_pred[2], y_pred[5]]
    # mu, s, a = y_pred
    s = np.log(1 + np.exp(s))
    # print mu, ' ', s, ' ', a

    from numpy.linalg import inv
    prob = np.abs(np.matmul(inv(np.diag(s)), np.asarray(y - mu)))
    print mu, ' ', s, ' ', a ,' -> ', y, ' has prob: ', prob

    id += 1
    if (id > 25):
        break

# Saving model
inps.save('/data/neuro_phys_sim/data/modelu2.h5')  # creates a HDF5 file
print('Saved model!')

print('Processed everything!')
