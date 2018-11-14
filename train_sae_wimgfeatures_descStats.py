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
    NME_nxgraphs = r'Z:\Cristina\Section3\breast_MR_NME_biological\NMEs_SER_nxgmetrics'
    
    allNMEs_dynamic = pd.read_csv(os.path.join(NME_nxgraphs,'dyn_roi_records_allNMEs_descStats.csv'), index_col=0)
       
    allNMEs_morphology = pd.read_csv(os.path.join(NME_nxgraphs,'morpho_roi_records_allNMEs_descStats.csv'), index_col=0)
    
    allNMEs_texture = pd.read_csv(os.path.join(NME_nxgraphs,'text_roi_records_allNMEs_descStats.csv'), index_col=0)
    
    allNMEs_stage1 = pd.read_csv(os.path.join(NME_nxgraphs,'stage1_roi_records_allNMEs_descStats.csv'), index_col=0)
                    
    # to load SERw matrices for all lesions
    with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_allNMEs_descStats.pklz'), 'rb') as fin:
        nxGdatafeatures = pickle.load(fin)
    
    # to load discrall_dict dict for all lesions
    with gzip.open(os.path.join(NME_nxgraphs,'nxGnormfeatures_allNMEs_descStats.pklz'), 'rb') as fin:
        discrall_dict_allNMEs = pickle.load(fin)           
        
    #########
    # shape input (798L, 427L)     
    nxGdiscfeatures = discrall_dict_allNMEs   
    print('Loading {} leasions with nxGdiscfeatures of size = {}'.format(nxGdiscfeatures.shape[0], nxGdiscfeatures.shape[1]) )
    
    print('Normalizing dynamic {} leasions with features of size = {}'.format(allNMEs_dynamic.shape[0], allNMEs_dynamic.shape[1]))
    normdynamic = (allNMEs_dynamic - allNMEs_dynamic.mean(axis=0)) / allNMEs_dynamic.std(axis=0)
    normdynamic.mean(axis=0)
    print(np.min(normdynamic, 0))
    print(np.max(normdynamic, 0))
    
    print('Normalizing morphology {} leasions with features of size = {}'.format(allNMEs_morphology.shape[0], allNMEs_morphology.shape[1]))
    normorpho = (allNMEs_morphology - allNMEs_morphology.mean(axis=0)) / allNMEs_morphology.std(axis=0)
    normorpho.mean(axis=0)
    print(np.min(normorpho, 0))
    print(np.max(normorpho, 0))
     
    print('Normalizing texture {} leasions with features of size = {}'.format(allNMEs_texture.shape[0], allNMEs_texture.shape[1]))
    normtext = (allNMEs_texture - allNMEs_texture.mean(axis=0)) / allNMEs_texture.std(axis=0)
    normtext.mean(axis=0)
    print(np.min(normtext, 0))
    print(np.max(normtext, 0))
    
    print('Normalizing stage1 {} leasions with features of size = {}'.format(allNMEs_stage1.shape[0], allNMEs_stage1.shape[1]))
    normstage1 = (allNMEs_stage1 - allNMEs_stage1.mean(axis=0)) / allNMEs_stage1.std(axis=0)
    normstage1.mean(axis=0)
    print(np.min(normstage1, 0))
    print(np.max(normstage1, 0))    
    
    # shape input (798L, 427L)    
    combX_allNME = np.concatenate((nxGdiscfeatures, normdynamic.as_matrix(), normorpho.as_matrix(), normtext.as_matrix(), normstage1.as_matrix()), axis=1)       
    YnxG_allNME = np.asarray([nxGdatafeatures['roi_id'].values,
            nxGdatafeatures['classNME'].values,
            nxGdatafeatures['nme_dist'].values,
            nxGdatafeatures['nme_int'].values])
            
    print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
    print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0])   )
     
                   
    ######################
    ## 2) Pre-train/fine tune the SAE
    ######################
    # set to INFO to see less information during training
    save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels'
    #logging.basicConfig(level=logging.DEBUG) 
    logging.basicConfig(filename=os.path.join(save_to,'train_SAE_wimgfeatures_descStats_zeromean.log'), 
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG) 
                        
    input_size = combX_allNME.shape[1]
    latent_size = [input_size/rxf for rxf in [25,15,10,5,2]]
    
    # train/test splits (test is 10% of labeled data)
    sep = int(combX_allNME.shape[0]*0.10)
    X_val = combX_allNME[:sep]
    y_val = YnxG_allNME[1][:sep]
    X_train = combX_allNME[sep:]
    y_train = YnxG_allNME[1][sep:]
    batch_size = 125 # 160 32*5 = update_interval*5
    X_val[np.isnan(X_val)] = 0.00001
    
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
                   'output_size': output_size,
                   'sep': sep}
                   
        allAutoencoders.append(outdict)
        # to save                      
        ae_model.save( os.path.join(save_to,'SAE_zsize{}_wimgfeatures_descStats_zeromean.arg'.format(output_size)) )
        logging.log(logging.INFO, "finished training and saving Autoencoder..: ")
        
    # save output
    allAutoencoders_file = gzip.open(os.path.join(save_to,'allAutoencoders_wimgfeatures_descStats_zeromean_log.pklz'), 'wb')
    pickle.dump(allAutoencoders, allAutoencoders_file, protocol=pickle.HIGHEST_PROTOCOL)
    allAutoencoders_file.close()
    ## to load
#    with gzip.open(os.path.join(save_to,'allAutoencoders_wimgfeatures_log.pklz'), 'rb') as fin:
#        allAutoencoders = pickle.load(fin)

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
                               
                               
    ######################
    ## Visualize the reconstructed inputs and the encoded representations. (AFTER TRINIGN)
    ######################
    ## to load
    allAutoencoders = []
    for output_size in latent_size:
        xpu = mx.cpu()
        ae_model = AutoEncoderModel(xpu, [X_train.shape[1],500,500,2000,output_size], pt_dropout=0.2)
        ae_model.load(os.path.join(save_to,'SAE_zsize{}_wimgfeatures_descStats_zeromean.arg'.format(output_size)))
        
         # put useful metrics in a dict
        outdict = {'E_train': ae_model.eval(X_train),
                   'E_val': ae_model.eval(X_val),
                   'output_size': output_size,
                   'sep': sep}
                   
        allAutoencoders.append(outdict)

