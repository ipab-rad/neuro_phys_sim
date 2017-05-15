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

# Params
NO_VALUE_IN_BIN = -1

# Load models
inps = mc.create_model('/data/neuro_phys_sim/data/modelu2.h5')

def draw_binned_sigma(bin_means, bin_sep, bins, id="", thr=1.0):
    bins_arrx = []
    bins_arry = []
    for bv, b in zip(bin_sep, bins):
        for v in bv:
            if (v != 0):
                bins_arrx.append(b)
                bins_arry.append(v)

    full_bins_idx = bin_means != NO_VALUE_IN_BIN
    # print ';IDX:', full_bins_idx, '\n', bin_means[full_bins_idx], '\n', bins[full_bins_idx]
    color = np.asarray([item < thr for item in bin_means[full_bins_idx]], dtype='int32')*0.7+0.2
    plt.plot(bins_arrx, bins_arry, 'r.', zorder=1, label='Standard deviations')
    plt.scatter(bins[full_bins_idx], bin_means[full_bins_idx],
                c=color, s=10, cmap='winter',
                zorder=10, label=u'Mean std per bin')
    plt.axhline(y = thr, color='g', zorder=11, label=u'1 std')
    print 'bin_means different from 0 - find: ', bin_means[full_bins_idx]
    std_means = bin_means[full_bins_idx].mean()
    plt.axhline(y = std_means, color='c', zorder=10, label=u'mean std ('+"{:.2f}".format(std_means)+')')
    plt.title('Histogram of mean std.')
    plt.xlim(-0.6, 0.6)
    plt.xlabel('delta x')
    plt.ylabel('std')
    plt.legend(loc='upper left')
    filename = "/data/neuro_phys_sim/images/mean_hist_"+str(epoch)+".png"
    plt.savefig(filename)
    plt.clf()
    print 'Saved figure - ', filename

def data2model(data):
    # model_out =  mc.pred_model(inps, data)
    model_out =  mc.pred_model(inps, data)
    model_out = np.asarray(model_out)
    return [model_out[0][0][0], model_out[1][0][0]], [model_out[0][0][1], model_out[1][0][1]] # get the means values

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
    print 'Updating extra archive'
    size, _ = sw.updateArchiveDirectly('/data/neuro_phys_sim/data/extra_data.hdf5', extra_crops, extra_outs)
    if (len(X) == 0 or len(Y) == 0):
        print 'Reading from dataset.'
        X, Y = sw.getDataFromArchive('/data/neuro_phys_sim/data/data_300.hdf5', sample_from_data=False)
        extra_crops, extra_outs = sw.getDataFromArchive('/data/neuro_phys_sim/data/extra_data.hdf5')

    X = np.concatenate((X, extra_crops), axis=0)
    Y = np.concatenate((Y, extra_outs), axis=0)

    print 'Total training data: ', X.shape

    Xc = mc.conv_input_batch_to_model(X)
    Yc = mc.conv_output_batch_to_model(Y)

    # Generate model
    batch_size = 256*8
    nb_epoch = 7
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, verbose=1,
                                  patience=2, cooldown=15, min_lr=0.000001)

    optimizer = Adamax(lr=1e-4, clipnorm=1e-6, clipvalue=1e6)
    inps.compile(optimizer=optimizer, loss=mc.mean_log_Gaussian_like)
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
    print '-----------'
    print 'Lead epoch: ', epoch

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

    print 'All delta x size: ', x_all.shape
    print 'All delta x prediction size: ', x_pred_all.shape
    print 'sigma_prob_all prediction size: ', sigma_prob_all.shape

    # Draw data
    draw_histograms(x_all[:,0], "eval_x"+str(epoch))
    draw_histograms(x_pred_all[:,0], "eval_x_pred"+str(epoch))
    draw_histograms(sigma_prob_all[:,0], "eval_sigma_prob_"+str(epoch), range=None)

    print('Processed evaluating MCMC chains!')

    # Format data
    x = x_all[:,0].reshape(-1, 1)
    y = x_pred_all[:,0].reshape(-1,)
    sigmas = sigma_prob_all[:,0].reshape(-1,)

    # Find mean stds of binned preductions
    bins, bin_size = np.linspace(np.min(x), np.max(x), num=50,
                                 endpoint=False, retstep=True)
    bins = np.asarray(bins)
    digitized = np.digitize(y, bins)
    bin_sep = np.asarray([sigmas[digitized == i] for i in range(0, len(bins))])
    bin_means = np.asarray([bin.mean() if not bin.size == 0 else NO_VALUE_IN_BIN
                 for bin in bin_sep])


    # Draw histogram of std
    draw_binned_sigma(bin_means, bin_sep, bins, epoch)

    # Sample most uncertain locations in log space
    # log_mean = np.log(bin_means)
    # log_mean = np.asarray([m if np.isfinite(m) else 0 for m in log_mean])
    # csum = np.cumsum(log_mean)
    csum = np.cumsum(bin_means)

    # print 'CSUM: ', csum, log_mean
    # if (csum[-1] <= 0):
    #     print 'Empty csum!'
    #     continue

    samples_from_histogram = 3
    samples_from_fitted_world = 3
    for x in xrange(samples_from_histogram):
        draw = np.random.uniform(low=0.0, high=csum[-1])
        print 'Draw: ', draw
        idxs = csum > draw
        # print 'Idxs ', idxs

        idx = next((i for i in enumerate(csum) if i[1] > draw), None)[0]
        # print idx, csum[idx]
        print 'Uncertainty mean:', bin_means[idx]
        print 'Bins: ', bins[idx], bins[idx]+bin_size
        bin_center = bins[idx] + bin_size/2.0


        _, diff_samples = s.mhmc(1000, prop_sigma=1, llh_func=get_x_pos_LLH)
        # print 'diff_samples ', len(diff_samples)

        for i in xrange(1, min(len(diff_samples), samples_from_fitted_world) + 1):
            x, x_pred, sigma_prob, crops, outs = sw.simulateWithModel(
                                                    diff_samples[-i], data2model, # TODO: Check if i is the correct size if samples_from_fitted_world is used
                                                    threshold_sigma=0.5)
            additional_training_crops = np.concatenate(
                                (additional_training_crops , np.asarray(crops)))
            additional_training_outs = np.concatenate(
                                (additional_training_outs , np.asarray(outs)))
            # print 'Addeed data shapes: ', additional_training_crops.shape, additional_training_outs.shape
            print 'Extra data ', len(crops)

        additional_training_crops = np.asarray(additional_training_crops)
        additional_training_outs = np.asarray(additional_training_outs)

    print 'Total Extra Data ', additional_training_crops.shape, additional_training_outs.shape
    print 'Retraining for lead epoch ', epoch
    inps, X, Y = retrain_network_with_new_samples(inps, additional_training_crops,
                                                additional_training_outs, X, Y)

    print 'Total training data now is: ', X.shape

    # Saving model
    inps.save('/data/neuro_phys_sim/data/models/model_refited_' + str(epoch) + '.h5')  # creates a HDF5 file
    print('Saved model!')

print 'Processing done.'
