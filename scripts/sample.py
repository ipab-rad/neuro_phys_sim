#!/usr/bin/env python

import numpy as np
# MPL no display backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sim_world as sw

from pymc3 import NUTS, sample
from scipy import optimize
from pymc3 import Model, Normal, HalfNormal, Slice
from pymc3 import summary
from pymc3 import traceplot

def draw_histograms(normv, id=0, range=0.6, draw_sorted=False):
    # Save to histogram
    plt.hist(normv, bins=50, facecolor='red')  # plt.hist passes it's arguments to np.histogram
    plt.title("x histogram with 50 bins")
    filename = "/data/neuro_phys_sim/images/hist_" + str(id) + ".png"
    if (range):
        plt.xlim(xmax = range, xmin = -range)
    plt.xlabel('delta x')
    plt.ylabel('occurances')
    plt.savefig(filename)
    plt.clf()
    print 'Saved figure - ', filename

    if draw_sorted:
        plt.plot(sorted(normv))
        plt.title("x sorted")
        filename = "/data/neuro_phys_sim/images/sorted_" + str(id) + ".png"
        if (range):
            plt.ylim(ymax = range, ymin = -range)
        plt.savefig(filename)
        plt.clf()
        print 'Saved figure - ', filename

# find likelihood of the magnitutde of v
def llh_data_np_nonnorm(data, mean, sigma):
    '''
    Based on http://arogozhnikov.github.io/2015/09/08/SpeedBenchmarks.html
    '''
    s = (data - mean) ** 2 / (2 * (sigma ** 2))
    pdfs = np.exp(- s)
    pdfs /= np.sqrt(2 * np.pi) * sigma # Normalization constant

    pdfs.sort()
    return np.log(pdfs).sum()

def testE2Esim():
    position, orientation, velocity, _, collisions_count, _ = sw.simulateWorld(
                                                    [1, 2, -0.8, 2, 3, 0.2, 0.5, 0, 0, 1.0,
                                                    0.3, 1.5],
                                                    saveVideo=False)

    # print position.shape, orientation.shape, velocity.shape, collisions_count
    normv = np.linalg.norm(velocity, axis = 1)
    # draw_histograms(normv, id=1)
    print 'llh: ', llh_data_np_nonnorm(normv, 1, 1)
    print 'Test passed.'

testE2Esim()

def th(x):
    return x

def get_vel_LLH(x):
    _, _, vel, dist2obj, c_count, impV = sw.simulateWorld(th(x), sim_length_s=5)
    normv = np.linalg.norm(vel, axis = 1)
    # print impV
    impVnorm = np.linalg.norm(impV, axis = 1)
    # target v
    # return llh_data_np_nonnorm(normv, target_mu, target_sigma) + \
    #        llh_data_np_nonnorm(dist2obj, 0, target_sigma) + \
    #        llh_data_np_nonnorm(impV[:,0], 1, target_sigma) + \
    #        llh_data_np_nonnorm(impVnorm, 5, 2)
    # return llh_data_np_nonnorm(dist2obj, 0, 0.1) # closest distance
    return llh_data_np_nonnorm(impVnorm, 2, 1) + \
           llh_data_np_nonnorm(dist2obj, 0, 1)
    # return llh_data_np_nonnorm(normv, target_mu, target_sigma) + \
    #        llh_data_np_nonnorm(dist2obj, 0, 1) # closest distance

def apply_range(value, vmin, vmax):
    if (value > vmax):
        value = vmax
    if (value < vmin):
        value = vmin
    return value

def create_urandom_sample():
    new_sample = np.zeros(12)
    new_sample[0] = np.random.uniform( -100, 100) # Impulse x
    new_sample[1] = np.random.uniform( -100, 100) # Impulse y
    new_sample[2] = np.random.uniform( -11, 11)   # paddle pos x
    new_sample[3] = np.random.uniform(   3, 13)   # paddle pos y
    new_sample[4] = np.random.uniform(-np.pi, np.pi) # paddle angle
    new_sample[5] = np.random.uniform(-np.pi, np.pi) # object angle
    new_sample[6] = np.random.uniform( .1, .90)  # object restitution
    new_sample[7] = np.random.uniform( -11, 11)   # obj x
    new_sample[8] = np.random.uniform(   3, 13)   # obj y
    new_sample[9] = np.random.uniform( 0.5, 2.5)  # obj density
    new_sample[10] = np.random.uniform( .1, .90)  # parent object restitution
    new_sample[11] = np.random.uniform( 0.5, 2.5)  # parent obj density
    return new_sample

def generate_sample_prop(old_sample):
    new_sample = old_sample + \
        np.concatenate([np.random.normal(0, 1, 2),
                    np.random.normal(0, 0.5, 2),
                    np.random.normal(0, 0.1, 2),
                    np.random.normal(0, 0.05, 1),
                    np.random.normal(0, 0.5, 2),
                    np.random.normal(0, 0.1, 1),
                    np.random.normal(0, 0.05, 1),
                    np.random.normal(0, 0.1, 1)]).flatten()

    # Apply constraints
    new_sample[0] = apply_range(new_sample[0], -100, 100)
    new_sample[1] = apply_range(new_sample[1], -100, 100)
    new_sample[2] = apply_range(new_sample[2], -11, 11)
    new_sample[3] = apply_range(new_sample[3],  3, 13)
    new_sample[4] = apply_range(new_sample[4],-np.pi, np.pi)
    new_sample[5] = apply_range(new_sample[5],-np.pi, np.pi)
    new_sample[6] = apply_range(new_sample[6], .1, .90)
    new_sample[7] = apply_range(new_sample[7], -11, 11)
    new_sample[8] = apply_range(new_sample[8],  3, 13)
    new_sample[9] = apply_range(new_sample[9], 0.5, 2.5)
    new_sample[10] = apply_range(new_sample[10], .1, .90)
    new_sample[11] = apply_range(new_sample[11], 0.5, 2.5)
    return new_sample

target_mu = 2
target_sigma = 1

# TODO: Make a function to sample k parallel mhmc chains
def mhmc(n, prop_sigma=1, llh_func=get_vel_LLH):
    # x = generate_sample_prop(np.zeros(10))
    x = create_urandom_sample()
    samples = np.zeros((n, len(x)))
    diff_samples = []
    # print x

    llh_old = llh_func(x)
    accepted_count = 0

    for i in xrange(n):
        x_new = generate_sample_prop(x)#x + np.random.randn(10) * prop_sigma
        # x_new = create_urandom_sample()
        llh_new = llh_func(x_new)

        if (i%100 == 0 and i != 0):
            print i, '/', n
            print 'llh new: ', llh_new, 'llh old: ', llh_old

        alpha = min(1, np.exp(llh_new - llh_old))
        if np.random.uniform(0, 1) <= alpha:
            if (llh_new != llh_old): # llh_new > -2000 and
                diff_samples.append(x_new)
            x = x_new
            accepted_count += 1
            llh_old = llh_new
            # print 'Accepted, new llh ', llh_new
        samples[i] = x
    print 'Accepted ratio:', float(accepted_count)/n
    print 'accepted_count: ', accepted_count
    return samples, diff_samples

if __name__ == "__main__":
    # samples, _ = mhmc(200, 1)
    # sw.simulateWorld(th(samples[-1]), saveVideo=True, filename="best.mp4")
    # print 'done'; exit(0)

    sample_attempts = 125
    for i in range(sample_attempts):
        print '-----------------------------'
        print 'Sampling ', i, ' / ', sample_attempts
        samples, diff_samples = mhmc(50, 1)
        print 'Done sampling.'
        # data, outdata = sw.generateData(th(samples[-1]))
        # _, _, vel, _, _, _ = sw.simulateWorld(th(samples[-1]), saveVideo=True, filename="best.mp4")
        # sw.simulateWorld(th(samples[5000]), saveVideo=True, filename="after_burn.mp4")

        # Save videos
        print 'Different samples len ', len(diff_samples)
        if len(diff_samples) == 0:
            continue
        id=0
        for s in reversed(diff_samples): # get best 10 samples
            # _, _, vel, _, _, _ = sw.simulateWorld(th(s), saveVideo=True, filename="list/after_burn"+str(i)+"_"+str(id)+".mp4")
            data, outdata = sw.generateData(th(s))
            # import cv2
            # cv2.imwrite("/data/neuro_phys_sim/data/cid.png", np.asarray(data['cid'][-1]*255))
            # cv2.imwrite("/data/neuro_phys_sim/data/scrop.png", np.asarray(data['scrop'][-1]*255))
            # cv2.imwrite("/data/neuro_phys_sim/data/mass.png", np.asarray(data['mass'][-1]*255))
            # cv2.imwrite("/data/neuro_phys_sim/data/velx.png", np.asarray(data['velx'][-1]*255))
            # cv2.imwrite("/data/neuro_phys_sim/data/vely.png", np.asarray(data['vely'][-1]*255))
            # cv2.imwrite("/data/neuro_phys_sim/data/avel.png", np.asarray(data['avel'][-1]*255))
            # cv2.imwrite("/data/neuro_phys_sim/data/rest.png", np.asarray(data['rest'][-1]*255))
            size, _ = sw.updateArchive('/data/neuro_phys_sim/data/data_300.hdf5', data, outdata)
            id += 1
            if (id >= 10):
                break
        diff_samples = []
        print "Dataset size: ", size
