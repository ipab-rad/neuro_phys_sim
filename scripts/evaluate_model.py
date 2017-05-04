import numpy as np

from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adamax

import sim_world as sw
import model_creation as mc

import sample as s
from sample import create_urandom_sample, draw_histograms, mhmc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load models
inps, m2 = mc.create_model('/data/neuro_phys_sim/data/model.h5')

def draw_sigma(bin_means, bin_sep, bins):
    bins_arrx = []
    bins_arry = []
    for bv, b in zip(bin_sep, bins):
        for v in bv:
            if (v != 0):
                bins_arrx.append(b)
                bins_arry.append(np.log(v))

    color = np.asarray([item < 3 for item in bin_means], dtype='uint8')*0.7+0.2
    plt.plot(bins_arrx, bins_arry, 'r.', zorder=1)
    plt.scatter(bins, np.log(bin_means), c=color, s=10, cmap='winter', zorder=10)
    plt.axhline(y = np.log(3), color='g', zorder=11)
    plt.title("Histogram of log mean std.")
    plt.xlim(-0.6, 0.6)
    filename = "/data/neuro_phys_sim/images/mean_hist.png"
    plt.savefig(filename)
    plt.clf()

def data2model(data):
    # model_out =  mc.pred_model(inps, data)
    model_out =  mc.pred_model(m2, data)

    return [model_out[0], model_out[2]], [model_out[1], model_out[3]] # get the means values

def get_x_pos_LLH(w):
    target_x = bin_center
    target_sigma = np.sqrt(bin_size)*3.0
    # print 'mean: ', target_x, 'sigma: ', target_sigma
    pos, _, vel, dist2obj, c_count, impV = sw.simulateWorld(s.th(w), sim_length_s=5)
    return s.llh_data_np_nonnorm(pos[:,0], target_x, target_sigma) + \
           s.llh_data_np_nonnorm(dist2obj, 0, 1)


def retrain_network_with_new_samples(model, extra_crops,
                                            extra_outs,
                                            X, Y):
    size, _ = sw.updateArchiveDirectly('/data/neuro_phys_sim/data/extra_data.hdf5', extra_crops, extra_outs)
    if (len(X) == 0 or len(Y) == 0):
        print 'Reading from dataset.'
        X, Y = sw.getDataFromArchive('/data/neuro_phys_sim/data/data.hdf5', sample_from_data=False)
        extra_crops, extra_outs = sw.getDataFromArchive('/data/neuro_phys_sim/data/extra_data.hdf5')

    X = np.concatenate((X, extra_crops), axis=0)
    Y = np.concatenate((Y, extra_outs), axis=0)

    Xc = mc.conv_input_batch_to_model(X)
    Yc = mc.conv_output_batch_to_model(Y)

    # Generate model
    batch_size = 256*8
    nb_epoch = 30
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, verbose=1,
                                  patience=5, cooldown=10, min_lr=0.000001)
    optimz = Adamax(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    inps.compile(optimizer=optimz, loss='mse')
    # Full train
    hist = inps.fit(Xc, Yc,
        batch_size=batch_size,
        epochs=nb_epoch,
        shuffle=True,  # does not affect valdiation set
        validation_split=0.1,
        callbacks=[reduce_lr]
        )
    return inps, X, Y

# All of the data
X = np.empty(shape=(0, 7, 32, 32))
Y = np.empty(shape=(0, 6))


total_epochs = 100
for epoch in xrange(total_epochs):
    x_all = np.empty(shape=(0, 2))
    x_pred_all = np.empty(shape=(0, 2))
    sigma_prob_all = np.empty(shape=(0, 2))

    additional_training_crops = np.empty(shape=(0, 7, 32, 32))
    additional_training_outs = np.empty(shape=(0, 6))

    for i in xrange(3):
        samples, _ = mhmc(1, 1)
        x, x_pred, sigma_prob, crops, outs = sw.simulateWithModel(samples[-1], data2model)
        # print "Add new values len: ", len(crops)
        x_all = np.concatenate((x_all, np.asarray(x)))
        x_pred_all = np.concatenate((x_pred_all, np.asarray(x_pred)))
        sigma_prob_all = np.concatenate((sigma_prob_all, np.asarray(sigma_prob)))
        # print 'Before Addeed data shapes: ', additional_training_crops.shape, additional_training_outs.shape
        additional_training_crops = np.concatenate((additional_training_crops, np.asarray(crops)), axis=0)
        additional_training_outs = np.concatenate((additional_training_outs, np.asarray(outs)), axis=0)
        # print 'Addeed data shapes: ', additional_training_crops.shape, additional_training_outs.shape

    sigma_prob_all = np.asarray(sigma_prob_all)

    print x_all.shape
    print x_pred_all.shape

    print 'All delta x size: ', x_all.shape
    print 'All delta x prediction size: ', x_pred_all.shape
    print 'sigma_prob_all prediction size: ', sigma_prob_all.shape

    #temp fix things
    # sigma_prob_all[sigma_prob_all > 1e3] = 1e3


    draw_histograms(x_all[:,0], "eval_x")
    draw_histograms(x_pred_all[:,0], "eval_x_pred")
    draw_histograms(sigma_prob_all[:,0], "eval_sigma_prob", range=None)

    print('Processed everything!')

    # Format data
    x = x_all[:,0].reshape(-1, 1)
    y = x_pred_all[:,0].reshape(-1,)
    sigmas = sigma_prob_all[:,0].reshape(-1,)

    # Find mean stds of binned preductions
    bins, bin_size = np.linspace(np.min(x), np.max(x), num=50,
                                 endpoint=False, retstep=True)
    digitized = np.digitize(y, bins)
    bin_sep = [sigmas[digitized == i] for i in range(0, len(bins))]
    bin_means = [bin.mean() if not bin.size == 0 else 0 for bin in bin_sep]

    # Draw histogram of std
    draw_sigma(bin_means, bin_sep, bins)

    # Sample most uncertain locations in log space
    log_mean = np.log(bin_means)
    log_mean = np.asarray([m if np.isfinite(m) else 0 for m in log_mean])
    csum = np.cumsum(log_mean)

    samples_from_histogram = 10
    samples_from_fitted_world = 10
    for x in xrange(samples_from_histogram):
        draw = np.random.uniform(low=0.0, high=csum[-1])
        # print 'Draw: ', draw
        idxs = csum > draw
        # print 'Idxs ', idxs

        idx = next((i for i in enumerate(csum) if i[1] > draw), None)[0]
        # print idx, csum[idx]
        print 'Uncertainty mean:', bin_means[idx]
        print 'Bins: ', bins[idx], bins[idx]+bin_size
        bin_center = bins[idx] + bin_size/2.0


        _, diff_samples = s.mhmc(1000, prop_sigma=1, llh_func=get_x_pos_LLH)
        # print 'diff_samples ', len(diff_samples)

        for i in xrange(min(len(diff_samples), samples_from_fitted_world)):
            x, x_pred, sigma_prob, crops, outs = sw.simulateWithModel(
                                                    diff_samples[-1], data2model,
                                                    threshold_sigma=0.5)
            additional_training_crops = np.concatenate((additional_training_crops , np.asarray(crops)))
            additional_training_outs = np.concatenate((additional_training_outs , np.asarray(outs)))
            # print 'Addeed data shapes: ', additional_training_crops.shape, additional_training_outs.shape
            print 'Extra data ', len(crops)

        additional_training_crops = np.asarray(additional_training_crops)
        additional_training_outs = np.asarray(additional_training_outs)

    print 'Total Extra Data ', additional_training_crops.shape, additional_training_outs.shape
    inps, X, Y = retrain_network_with_new_samples(inps, additional_training_crops,
                                                additional_training_outs, X, Y)


    print 'Total training data now is: ', X.shape


# # =======
# print 'Checking improvements'
# x_all = np.empty(shape=(0, 2))
# x_pred_all = np.empty(shape=(0, 2))
# sigma_prob_all = np.empty(shape=(0, 2))
# x, x_pred, sigma_prob, crops, outs = sw.simulateWithModel(samples[-1], data2model)
# print "Add new values len: ", len(crops)
# x_all = np.concatenate((x_all, np.asarray(x)))
# x_pred_all = np.concatenate((x_pred_all, np.asarray(x_pred)))
# sigma_prob_all = np.concatenate((sigma_prob_all, np.asarray(sigma_prob)))

# draw_histograms(x_all[:,0], "updated_eval_x")
# draw_histograms(x_pred_all[:,0], "updated_eval_x_pred")
# draw_histograms(sigma_prob_all[:,0], "updated_eval_sigma_prob", range=None)

# x = x_all[:,0].reshape(-1, 1)
# y = x_pred_all[:,0].reshape(-1,)
# sigmas = sigma_prob_all[:,0].reshape(-1,)

# # Find mean stds of binned preductions
# bins, bin_size = np.linspace(np.min(x), np.max(x), num=50,
#                              endpoint=False, retstep=True)
# digitized = np.digitize(y, bins)
# bin_sep = [sigmas[digitized == i] for i in range(0, len(bins))]
# bin_means = [bin.mean() if not bin.size == 0 else 0 for bin in bin_sep]

# # Draw histogram of std
# draw_sigma(bin_means, bin_sep, bins)

# Saving model
    inps.save('/data/neuro_phys_sim/data/model_refited.h5')  # creates a HDF5 file
    print('Saved model!')

print 'Done'
# ############## USELESS GPs

# # Fit GP
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# kernel =  RBF(length_scale=0.1, length_scale_bounds=(1e-5, 1e2)) + \
#           WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e+1))#* ConstantKernel(1.0, (1e-3, 1e3))
# # Ground truth data
# X = x_all[:,0].reshape(-1, 1)

# # Observations and noise
# y = x_pred_all[:,0].reshape(-1,)
# noise_variable = 0.0005

# print "x_all[:,0] extra shape", X.shape
# print "y_pred_all[:,0] extra shape", y.shape

# # Instanciate a Gaussian Process model
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
#                                 alpha=noise_variable )# normalize_y=True

# # Fit to data using Maximum Likelihood Estimation of the parameters
# gp.fit(X, y)

# # Make the prediction on the meshed x-axis (ask for MSE as well)
# # x = np.atleast_2d(np.linspace(np.min(X), np.max(X), 100)).T
# x = np.atleast_2d(np.linspace(-0.5, 0.5, 100)).T
# y_pred, sigma = gp.predict(x, return_std=True)

# # Plot the function, the prediction and the 95% confidence interval based on
# # the MSE
# fig = plt.figure()
# plt.plot(x, x, 'r:', label=u'$f(x) = x $')
# plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
# plt.plot(x, y_pred, 'b-', label=u'Prediction')
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([y_pred - 1.9600 * sigma,
#                         (y_pred + 1.9600 * sigma)[::-1]]),
#          alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$x$')
# plt.ylabel('$x_{hat}$')
# plt.ylim(np.min(X), np.max(X))
# plt.xlim(np.min(X), np.max(X))
# plt.legend(loc='upper left')
# plt.savefig('/data/neuro_phys_sim/data/gp.png')

# print 'Saved fig'
