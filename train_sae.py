# pylint: skip-file
import sys
import os
import mxnet as mx
import numpy as np
import pandas as pd
import data
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import model
from autoencoder import AutoEncoderModel
from solver import Solver, Monitor
import logging
import sklearn
from sklearn.manifold import TSNE
from utilities import *
try:
   import cPickle as pickle
except:
   import pickle
import gzip
   

if __name__ == '__main__':
    #####################################################
    ## 1) read in the datasets both all NME (to do pretraining)
    NME_nxgraphs = r'Z:\Cristina\Section3\NME_DEC\imgFeatures\NME_nxgraphs'
    # start by loading nxGdatafeatures
    with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_allNMEs_10binsize.pklz'), 'rb') as fin:
        nxGdatafeatures = pickle.load(fin)
    
    # to load discrall_dict dict for all lesions
    with gzip.open(os.path.join(NME_nxgraphs,'nxGnormfeatures_allNMEs_10binsize.pklz'), 'rb') as fin:
        discrall_dict_allNMEs = pickle.load(fin)           
        
    # shape input (798L, 427L)    
    combX_allNME = discrall_dict_allNMEs   
    YnxG_allNME = [nxGdatafeatures['roi_id'].values,
            nxGdatafeatures['classNME'].values,
            nxGdatafeatures['nme_dist'].values,
            nxGdatafeatures['nme_int'].values]
    
    print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
    print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0])   )
    
    ######################
    ## 2) Pre-train/fine tune the SAE
    ######################
    # set to INFO to see less information during training
    save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels'
    logging.basicConfig(filename=os.path.join(save_to,'train_SAE.log'), 
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG) 
                        
    input_size = combX_allNME.shape[1]
    latent_size = [input_size/rxf for rxf in [25,15,10,5]]
    
    # batch normalization
    sep = int(combX_allNME.shape[0]*0.75)
    X_train = combX_allNME[:sep]
    X_val = combX_allNME[sep:]
    batch_size = 125 # 
    
    allAutoencoders = []
    for output_size in latent_size:
        # Train or Read autoencoder: interested in encoding/decoding the input nxg features into LD latent space        
        # optimized for clustering with DEC
        xpu = mx.cpu()
        ae_model = AutoEncoderModel(xpu, [X_train.shape[1],500,500,2000,output_size], pt_dropout=0.2)
        ##  Pre-train
        ae_model.layerwise_pretrain(X_train, batch_size, 50000, 'sgd', l_rate=0.1, decay=0.0,
                                    lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
        ##  finetune
        ae_model.finetune(X_train, batch_size, 100000, 'sgd', l_rate=0.1, decay=0.0,
                      lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
                      
        ##  Get train/valid error (for Generalization)
        logging.log(logging.INFO, "Autoencoder Training error: %f"%ae_model.eval(X_train))
        logging.log(logging.INFO, "Autoencoder Validation error: %f"%ae_model.eval(X_val))
        # put useful metrics in a dict
        outdict = {'E_train': ae_model.eval(X_train),
                   'E_val': ae_model.eval(X_val),
                   'output_size': output_size}
        allAutoencoders.append(outdict)
        # to save                      
        ae_model.save( os.path.join(save_to,'SAE_zsize{}.arg'.format(output_size)) )
        logging.log(logging.INFO, "finished training and saving Autoencoder..: ")
        
    # save output
    allAutoencoders_file = gzip.open(os.path.join(save_to,'allAutoencoders_log.pklz'), 'wb')
    pickle.dump(allAutoencoders, allAutoencoders_file, protocol=pickle.HIGHEST_PROTOCOL)
    allAutoencoders_file.close()
    
    ######################
    ## Visualize the reconstructed inputs and the encoded representations.
    ######################
    # train/test loss value o
    dfSAE_perf = pd.DataFrame()
    for SAE_perf in allAutoencoders:
        dfSAE_perf = dfSAE_perf.append( pd.DataFrame({'Reconstruction Error': pd.Series(SAE_perf)[0:2], 'train/validation':pd.Series(SAE_perf)[0:2].index, 'compressed size': SAE_perf['output_size']}) ) 
        
    import seaborn as sns
    sns.set_style("darkgrid")
    axSAE_perf = sns.pointplot(x="compressed size", y="Reconstruction Error", hue="train/validation", data=dfSAE_perf,  
                               markers=["o", "x"], linestyles=["-", "--"])                               
