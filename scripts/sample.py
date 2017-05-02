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

def draw_histograms(normv, id=0):
    # Save to histogram
    plt.hist(normv, bins=50, facecolor='red')  # plt.hist passes it's arguments to np.histogram
    plt.title("Vel x histogram with 50 bins")
    filename = "images/hist_" + str(id) + ".png"
    plt.xlim(xmax = 0.6, xmin = -0.6)
    plt.savefig(filename)
    plt.clf()

    plt.plot(sorted(normv))
    plt.title("Vel x sorted")
    filename = "images/sorted_" + str(id) + ".png"
    plt.ylim(ymax = 0.6, ymin = -0.6)
    plt.savefig(filename)
    plt.clf()

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
    # return [x[2], x[3], x[4], x[6], x[7], x[5], x[0], x[1]]
    # return [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]]
    return x

def get_vel_LLH(x):
    _, _, vel, dist2obj, c_count, impV = sw.simulateWorld(th(x), sim_length_s=5)
    # if c_count == 0:
    #     return -np.inf
    normv = np.linalg.norm(vel, axis = 1)
    # print impV
    impVnorm = np.linalg.norm(impV, axis = 1)
     # target v
    # return llh_data_np_nonnorm(normv, target_mu, target_sigma) + \
    #        llh_data_np_nonnorm(dist2obj, 0, target_sigma) + \
    #        llh_data_np_nonnorm(impV[:,0], 1, target_sigma) + \
    #        llh_data_np_nonnorm(impVnorm, 5, 2)
    # return llh_data_np_nonnorm(dist2obj, 0, 0.1) # closest distance
    return llh_data_np_nonnorm(impVnorm, 10, 1) + \
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
    new_sample[2] = np.random.uniform( -12, 12)   # paddle pos x
    new_sample[3] = np.random.uniform(   2, 13)   # paddle pos y
    new_sample[4] = np.random.uniform(-np.pi, np.pi) # paddle angle
    new_sample[5] = np.random.uniform(-np.pi, np.pi) # object angle
    new_sample[6] = np.random.uniform( .1, .90)  # object restitution
    new_sample[7] = np.random.uniform( -12, 12)   # obj x
    new_sample[8] = np.random.uniform(   2, 13)   # obj y
    new_sample[9] = np.random.uniform( 0.5, 2.5)  # obj density
    new_sample[10] = np.random.uniform( .1, .90)  # parent object restitution
    new_sample[11] = np.random.uniform( 0.5, 2.5)  # parent obj density
    return new_sample

def generate_sample_prop(old_sample):
    # print
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
    new_sample[2] = apply_range(new_sample[2], -12, 12)
    new_sample[3] = apply_range(new_sample[3],  2, 13)
    new_sample[4] = apply_range(new_sample[4],-np.pi, np.pi)
    new_sample[5] = apply_range(new_sample[5],-np.pi, np.pi)
    new_sample[6] = apply_range(new_sample[6], .1, .90)
    new_sample[7] = apply_range(new_sample[7], -12, 12)
    new_sample[8] = apply_range(new_sample[8],  2, 13)
    new_sample[9] = apply_range(new_sample[9], 0.5, 2.5)
    new_sample[10] = apply_range(new_sample[10], .1, .90)
    new_sample[11] = apply_range(new_sample[11], 0.5, 2.5)
    return new_sample

target_mu = 2
target_sigma = 1

def mhmc(n, prop_sigma=1):
    # x = generate_sample_prop(np.zeros(10))
    x = create_urandom_sample()
    samples = np.zeros((n, len(x)))
    diff_samples = []
    # print x

    llh_old = get_vel_LLH(x)
    accepted_count = 0

    for i in xrange(n):
        x_new = generate_sample_prop(x)#x + np.random.randn(10) * prop_sigma
        # x_new = create_urandom_sample()
        llh_new = get_vel_LLH(x_new)

        if (i%50 == 0):
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
    # print 'done'

    for i in range(200):
        print '-----------------------------'
        samples, diff_samples = mhmc(50, 1)
        print 'Done sampling ', i
        # data, outdata = sw.generateData(th(samples[-1]))
        # _, _, vel, _, _, _ = sw.simulateWorld(th(samples[-1]), saveVideo=True, filename="best.mp4")
        # sw.simulateWorld(th(samples[5000]), saveVideo=True, filename="after_burn.mp4")

        # normv = np.linalg.norm(vel, axis = 1)
        # draw_histograms(normv, "best")

        # Save videos
        print 'different samples len ', len(diff_samples)
        id=0
        for s in reversed(diff_samples): # get best 10 samples
            # _, _, vel, _, _, _ = sw.simulateWorld(th(s), saveVideo=True, filename="list/after_burn"+str(i)+"_"+str(id)+".mp4")
            data, outdata = sw.generateData(th(samples[-1]))
            # import cv2
            # cv2.imwrite("data/cid.png", np.asarray(data['cid'][-1]*255))
            # cv2.imwrite("data/scrop.png", np.asarray(data['scrop'][-1]*255))
            # cv2.imwrite("data/mass.png", np.asarray(data['mass'][-1]*255))
            # cv2.imwrite("data/velx.png", np.asarray(data['velx'][-1]*255))
            # cv2.imwrite("data/vely.png", np.asarray(data['vely'][-1]*255))
            # cv2.imwrite("data/avel.png", np.asarray(data['avel'][-1]*255))
            # cv2.imwrite("data/rest.png", np.asarray(data['rest'][-1]*255))
            size, _ = sw.updateArchive('data/data.hdf5', data, outdata)
            id += 1
            if (id > 10):
                break
        diff_samples = []
        print "Dataset size: ", size
