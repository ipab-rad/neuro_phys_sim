import cv2
import numpy as np
import re
import glob
#files =  glob.glob("datatest/data*.npy") ## "/media/daniel/8a78d15a-fb00-4b6e-9132-0464b3a45b8b/simdata3/"
# files.sort()
files =  glob.glob("/media/daniel/8a78d15a-fb00-4b6e-9132-0464b3a45b8b/simdata3/data*.npy")

files = sorted(files)

bigdata = []
bigresp = []

for f in files:
    print(filter(None, re.split('[_.]', f)))
    ldir, seed, id, _ = filter(None, re.split('[_.]', f))
    data = np.asarray(np.load(f))
    resp = np.asarray(np.load(ldir[:-4] + 'resp_'+str(seed)+'_'+str(id)+'.npy'))
    bigdata.append(data)
    # bigresp.append(resp)
    vis = np.concatenate((data[:,:,0], data[:,:,1]), axis=1)
    vis = np.concatenate((vis, data[:,:,2]), axis=1)
    vis = np.concatenate((vis, data[:,:,3]), axis=1)
    vis = np.concatenate((vis, data[:,:,4]), axis=1)
    vis = np.concatenate((vis, data[:,:,5]), axis=1)
    vis = np.concatenate((vis, data[:,:,6]), axis=1)

    cv2.imshow("all_data_" + str(id), vis)
    # print resp
    # d = data[:,:,4]
    # print((d>0.01).sum())
    # print(d[np.where( d > 0.01 ) ])
    # print d
    # cv2.imshow("data", d)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

bigdata = np.asarray(bigdata)
# bigresp = np.asarray(bigresp)

print bigdata.shape
# print bigresp.shape
