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
nb_epoch = 20


inps, m2 = mc.create_model() # load the previously trained model '/data/neuro_phys_sim/data/model.h5'
# inps.compile(optimizer='adam', loss='mse') # , metrics=['accuracy']
# inps.compile(optimizer=optimz, loss='mse') # , metrics=['accuracy']
# m2.compile(optimizer='adam', loss=mc.likelihood_loss) # , metrics=['accuracy']

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, verbose=1,
                              patience=5, cooldown=10, min_lr=0.000001)

# Plot model
# inps.summary()
mc.plot_model(m2)

# Load data
X, Y = sw.getDataFromArchive('/data/neuro_phys_sim/data/data_300.hdf5', sample_from_data=False)

Xc = mc.conv_input_batch_to_model(X)
# Yc = mc.conv_output_batch_to_model(Y)

# draw_histograms(Yc[0], "x")
# draw_histograms(Yc[1], "y")
oval = np.asarray(Y[:,0]).reshape(-1, 1)

print oval.shape
Yc = np.ones(shape=(oval.shape[0], oval.shape[1]*3), dtype=np.float32) * 1e-5 * 0
# import random
# Yc = np.asarray([[random.random()*0.5 for i in range(3)] for j in range(oval.shape[0])])
Yc[:oval.shape[0], :oval.shape[1]] = oval
print Yc

print Xc[0].shape
print Yc[0].shape

# # Optimizing mean
# print 'Optimizing mean vector!'
# optimz_mean = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# m2.compile(optimizer=optimz_mean, loss='mse')

# # Full train
# hist = m2.fit(Xc, Yc,
#     batch_size=batch_size,
#     epochs=2,
#     shuffle=True,  # does not affect valdiation set
#     validation_split=0.1,
#     callbacks=[reduce_lr]
#     )

print 'Optimizing mean and variance with custom loss!'
optimizer = Adamax(clipnorm=1e-6, clipvalue=1e6)
m2.compile(optimizer=optimizer, loss=mc.mean_log_Gaussian_like)

# Full train
hist = m2.fit(Xc, Yc,
    batch_size=batch_size,
    epochs=nb_epoch,
    shuffle=True,  # does not affect valdiation set
    validation_split=0.1,
    callbacks=[reduce_lr]
    )
    # callbacks=[early_stopping])

# Loss and metrics
m2.compile(optimizer='adam', loss='mse')
loss_and_metrics = m2.evaluate(Xc, Yc, batch_size=batch_size, verbose=1)
print('Total loss: ' + str(loss_and_metrics))
# m2.compile(loss='mse', optimizer='sgd')
# loss_and_metrics = m2.evaluate(Xc, [Yc[0], np.zeros(Yc[0].shape),
#                                     Yc[1], np.zeros(Yc[1].shape)],
#                                     batch_size=batch_size, verbose=1)
# print('Total loss: ' + str(loss_and_metrics))

Yc_pred = m2.predict(Xc, batch_size=8*256)
id = 0
for y, y_pred in zip(Yc, Yc_pred):
    # Get probability
    mu, s, a = y_pred
    s = np.log(1 + np.exp(s))

    prob = np.abs(np.matmul(inv(np.diag(s)), np.asarray(y - mu)))
    print mu, ' ', s, ' ', a ,' -> ', y[0], ' has prob: ', prob
    # print 'Prob: ', prob

    id += 1
    if (id > 25):
        break

# Saving model
inps.save('/data/neuro_phys_sim/data/modelu.h5')  # creates a HDF5 file
print('Saved model!')

print('Processed everything!')
