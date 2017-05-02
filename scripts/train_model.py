import numpy as np
import time as t
from keras.models import Model, Sequential
from keras.layers import Input, Lambda
from keras.layers.merge import concatenate
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, Adamax
import sim_world as sw
import model_creation as mc
from sample import draw_histograms

# MPL no display backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# end

# Generate model
batch_size = 256*8
nb_epoch = 300


inps, m2 = mc.create_model() # load the previously trained model '/data/neuro_phys_sim/data/model.h5'
optimz = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
inps.compile(optimizer='adam', loss='mse') # , metrics=['accuracy']
# inps.compile(optimizer=optimz, loss='mse') # , metrics=['accuracy']
# inps.compile(optimizer=optimz, loss=mc.likelihood_loss) # , metrics=['accuracy']

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, verbose=1,
                              patience=5, cooldown=10, min_lr=0.000001)

# Plot model
# inps.summary()
mc.plot_model(inps)

# Load data
X, Y = sw.getDataFromArchive('/data/neuro_phys_sim/data/data.hdf5', sample_from_data=False)

Xc = mc.conv_input_batch_to_model(X)
Yc = mc.conv_output_batch_to_model(Y)

draw_histograms(Yc[0], "x")
draw_histograms(Yc[1], "y")

print Yc[0].shape
print Xc[0].shape

# Full train
hist = inps.fit(Xc, Yc,
    batch_size=batch_size,
    epochs=nb_epoch,
    shuffle=True,  # does not affect valdiation set
    validation_split=0.1,
    callbacks=[reduce_lr]
    )
    # callbacks=[early_stopping])

# Loss and metrics
loss_and_metrics = inps.evaluate(Xc, Yc, batch_size=batch_size, verbose=1)
print('Total loss: ' + str(loss_and_metrics))
m2.compile(loss='mse', optimizer='sgd')
loss_and_metrics = m2.evaluate(Xc, [Yc[0], np.zeros(Yc[0].shape),
                                    Yc[1], np.zeros(Yc[1].shape)],
                                    batch_size=batch_size, verbose=1)
print('Total loss: ' + str(loss_and_metrics))

# Saving model
inps.save('/data/neuro_phys_sim/data/model.h5')  # creates a HDF5 file
print('Saved model!')

print('Processed everything!')
