import os
import numpy as np
from sklearn.datasets import fetch_mldata
import six.moves.cPickle as pickle
import gzip

def get_mnist():
    np.random.seed(1234) # set seed for deterministic ordering
    data_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    data_path = os.path.join(data_path, 'datasets')
    if not os.path.exists(data_path):
        os.mkdir(data_path)        
    print"saving mnist to %s..."%data_path
    mnist = fetch_mldata('MNIST original', data_home=data_path)
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p].astype(np.float32)*0.02
    Y = mnist.target[p]
    return X, Y
    
    
def get_serw():
    np.random.seed(1234) # set seed for deterministic ordering
    data_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    data_path = os.path.join(data_path, 'datasets')
    with gzip.open(os.path.join(data_path,'allNMEs_25binsizenormSERcounts_probC_x.pklz'), 'rb') as fu:
        normSERcounts = pickle.load(fu)
    
    with gzip.open(os.path.join(data_path,'allNMEs_25binsizedatanormSERcounts_probC_x.pklz'), 'rb') as f:
        datanormSERcounts = pickle.load(f)
          
    print"loading discretized SER bins %s..."%data_path
    # set up some parameters and define labels
    X = np.asarray(normSERcounts)
    Y =  datanormSERcounts['class'].values
    print"X is mxn-dimensional with m = %i discretized SER bins" % X.shape[1]
    print"X is mxn-dimensional with n = %i cases" % X.shape[0]    
    return X, Y
    
    
def get_nxGfeatures():
    data_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    data_path = os.path.join(data_path, 'datasets')
    with gzip.open(os.path.join(data_path,'nxGdatafeatures_allNMEs_10binsize.pklz'), 'rb') as fu:
        nxGdatafeatures = pickle.load(fu)

    with gzip.open(os.path.join(data_path,'nxGdiscdatafeatures_allNMEs_10binsize.pklz'), 'rb') as fu:
        nxGdiscdatafeatures = pickle.load(fu)        
    
    with gzip.open(os.path.join(data_path,'nxGnormfeatures_allNMEs_10binsize.pklz'), 'rb') as fu:
        nxGnormfeatures = pickle.load(fu)
              
    print"loading nx graph features %s..."%data_path
    # set up some parameters and define labels
    X = nxGnormfeatures
    print"nxGdatafeatures is mxn-dimensional with m = %i nx graph features" % X.shape[1]
    print"nxGdatafeatures is mxn-dimensional with n = %i cases" % X.shape[0]
    y =  nxGdatafeatures['roiBIRADS'].values
    y0 =  nxGdatafeatures['classNME'].values
    y1 = nxGdatafeatures['roi_id'].values
    y2 = nxGdatafeatures['nme_dist'].values
    y3 = nxGdatafeatures['nme_int'].values
    y4 = nxGdatafeatures['dce_init'].values
    y5 = nxGdatafeatures['dce_delay'].values
    y6 = nxGdatafeatures['nme_t2si'].values
    Y = [y,y0,y1,y2,y3,y4,y5,y6]
    return X, Y, nxGdatafeatures
    
    