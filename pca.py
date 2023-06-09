#######################
### Import Packages ###
#######################

import sys
import numpy as np
import json
import time
import dataflow as flow
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

#####################
### Main Function ###
#####################

def main(args):

    ### parse input dictionary
    logfile = args['logfile']
    fly_idx = args['fly_idx']
    printlog = getattr(flow.Printlog(logfile=logfile), 'print_to_log')

    printlog('numpy: ' + str(np.__version__))


    load_file = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20210130_superv_depth_correction/super_brain.pickle'
    with open(load_file, 'rb') as handle:
        temp_brain = pickle.load(handle)
    #brain is a dict of z, each containing a variable number of supervoxels
    #one dict element looks like: (n_clusters, 3384, 9)
    X = np.zeros((0,3384,9))
    #for z in range(49):
    for z in range(9,49-9):
        X = np.concatenate((X,temp_brain[z]),axis=0)

    printlog(str(X.shape))
    X = np.swapaxes(X,1,2)
    X = np.reshape(X,(-1, 30456))
    X = X.T

    printlog('X is time by voxels {}'.format(X.shape))
    num_tp = 3384
    start = fly_idx*num_tp
    stop = (fly_idx+1)*num_tp
    X = X[start:stop,:] ### grab neural data for the right fly. 
    printlog(F'fly_idx: {fly_idx}, start: {start}, stop: {stop}')
    printlog('After fly_idx, X is time by voxels {}'.format(X.shape))

    printlog('Using np.linalg.ein')
    covariance_matrix = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    printlog('eigen_values is {}'.format(eigen_values.shape))
    save_file = F'/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20210130_superv_depth_correction/20210214_eigen_values_ztrim_fly{fly_idx}.npy'
    np.save(save_file, eigen_values)

    printlog('eigen_vectors is {}'.format(eigen_vectors.shape))
    save_file = F'/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20210130_superv_depth_correction/20210214_eigen_vectors_ztrim_fly{fly_idx}.npy'
    np.save(save_file, eigen_vectors)

if __name__ == '__main__':
    main(json.loads(sys.argv[1]))