# -*- coding: utf-8 -*-
"""
Created on Wed May 03 16:47:40 2017

@author: DeepLearning
"""

##############################################################################################################
# 1) load saved variables
##############################################################################################################
import sys
import os
import mxnet as mx
import numpy as np
import pandas as pd

from utilities import *
import data
import model
from autoencoder import AutoEncoderModel
from solver import Solver, Monitor
import logging

from sklearn.manifold import TSNE
from utilities import *
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import sklearn.neighbors 
import matplotlib.patches as mpatches
from sklearn.utils.linear_assignment_ import linear_assignment

try:
   import cPickle as pickle
except:
   import pickle
import gzip

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

#####################################################
from decModel_exp5_wimgF import *

## 1) read in the datasets both all NME (to do pretraining)
NME_nxgraphs = r'Z:\Cristina\Section3\NME_DEC\imgFeatures\NME_nxgraphs'

with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_dynamic.pklz'), 'rb') as fin:
    allNMEs_dynamic = pickle.load(fin)

with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_morphology.pklz'), 'rb') as fin:
    allNMEs_morphology = pickle.load(fin)        

with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_texture.pklz'), 'rb') as fin:
    allNMEs_texture = pickle.load(fin)

with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_stage1.pklz'), 'rb') as fin:
    allNMEs_stage1 = pickle.load(fin) 

# to load SERw matrices for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_allNMEs_10binsize.pklz'), 'rb') as fin:
    nxGdatafeatures = pickle.load(fin)

# to load discrall_dict dict for all lesions
with gzip.open(os.path.join(NME_nxgraphs,'nxGnormfeatures_allNMEs_10binsize.pklz'), 'rb') as fin:
    discrall_dict_allNMEs = pickle.load(fin)           

#########
# shape input (798L, 427L)    
nxGdiscfeatures = discrall_dict_allNMEs   
print('Loading {} all nxGdiscfeatures of size = {}'.format(nxGdiscfeatures.shape[0], nxGdiscfeatures.shape[1]) )
print(np.min(nxGdiscfeatures, 0))
print(np.max(nxGdiscfeatures, 0))

print 'Normalizing dynamic..  \n min={}, \n max={} \n'.format(np.min(allNMEs_dynamic, 0), np.max(allNMEs_dynamic, 0))
x_min, x_max = np.min(allNMEs_dynamic, 0), np.max(allNMEs_dynamic, 0)
x_max[x_max==0]=1.0e-07
normdynamic = (allNMEs_dynamic - x_min) / (x_max - x_min)

print 'Normalizing morphology..  \n min={}, \n max={} \n'.format(np.min(allNMEs_morphology, 0), np.max(allNMEs_morphology, 0))
x_min, x_max = np.min(allNMEs_morphology, 0), np.max(allNMEs_morphology, 0)
x_max[x_max==0]=1.0e-07
normorpho = (allNMEs_morphology - x_min) / (x_max - x_min)

print 'Normalizing texture..  \n min={}, \n max={} \n'.format(np.min(allNMEs_texture, 0), np.max(allNMEs_texture, 0))
x_min, x_max = np.min(allNMEs_texture, 0), np.max(allNMEs_texture, 0)
x_max[x_max==0]=1.0e-07
normtext = (allNMEs_texture - x_min) / (x_max - x_min)

print 'Normalizing stage1..  \n min={}, \n max={} \n'.format(np.min(allNMEs_stage1, 0), np.max(allNMEs_stage1, 0))
x_min, x_max = np.min(allNMEs_stage1, 0), np.max(allNMEs_stage1, 0)
x_min[np.isnan(x_min)]=1.0e-07
x_max[np.isnan(x_max)]=1.0
normstage1 = (allNMEs_stage1 - x_min) / (x_max - x_min)
normstage1[np.isnan(normstage1)]=1.0e-07

# shape input (798L, 427L)    
combX_allNME = np.concatenate((nxGdiscfeatures, normdynamic, normorpho, normtext, normstage1), axis=1)       
YnxG_allNME = np.asarray([nxGdatafeatures['roi_id'].values,
        nxGdatafeatures['classNME'].values,
        nxGdatafeatures['nme_dist'].values,
        nxGdatafeatures['nme_int'].values])

print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0])   )

## 1a) Load decModel_exp2 results
######################
labeltype = 'NME_dist_int_enh' # roilabel_NME_dist
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_exp2'
logging.basicConfig(level=logging.DEBUG)  
#logging.basicConfig(filename=os.path.join(save_to,'allDEC_exp2_y{}.log'.format(labeltype)), format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)                    

# dfine num_centers according to clustering variable
## use y_dec to  minimizing KL divergence for clustering with known classes
X = combX_allNME  
input_size = X.shape[1]
latent_size = [input_size/rxf for rxf in [25,15,10,5]]

ysup = YnxG_allNME[1]+'_'+YnxG_allNME[3] 
ysup = ['K'+rl[1::] if rl[0]=='U' else rl for rl in ysup]
roi_labels = YnxG_allNME[1]  
roi_labels = ['K' if rl=='U' else rl for rl in roi_labels]
num_centers = len(np.unique(ysup))
fig = plt.figure(figsize=(20,6))
ax = plt.axes()
plt.gca().set_color_cycle(['red', 'green', 'blue', 'cyan'])

# to load a prevously DEC model
dfDEC_perf = pd.DataFrame()
for znum in latent_size:
    # Read autoencoder: note is not dependent on number of clusters just on z latent size
    print "... loading a prevously DEC model of latent size znum = ",znum
    dec_model = DECModel(mx.cpu(), combX_allNME, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels')
    # to load best args found during DEC
    with gzip.open(os.path.join(save_to,'dec_model_znum{}_exp2_y{}.arg'.format(znum,labeltype)), 'rb') as fu:
        best_args = pickle.load(fu)
    # algorithm to find the epoch at best acc   
    max_acc = 0
    epochs = []; best_acc = []
    for k,acc in enumerate(best_args['acci']):
        if(acc >= max_acc):
            epochs.append(k)
            best_acc.append(acc)
            max_acc = acc
    # to plot best acci
    dfDEC_perf = dfDEC_perf.append( pd.DataFrame({'bestacci': pd.Series(best_acc), 'iterations':epochs, 'Z-spaceDim':znum }) )
    ax.plot(epochs, best_acc, '.-')
    
# add legend    
plt.legend([str(latsize) for latsize in latent_size], loc='upper left')
# find max
np.max(dfDEC_perf['bestacci'])
print  dfDEC_perf[dfDEC_perf['bestacci'] == np.max(dfDEC_perf['bestacci'])]
fig.savefig(save_to+os.sep+'allDEC_exp2_bestacci_y{} Dataset_unsuprv acc vs iteration.pdf'.format(labeltype), bbox_inches='tight')    
    
# save to R
#    pdzfinal = pd.DataFrame( np.append( y[...,None], zfinal, 1) )
#    pdzfinal.to_csv('datasets//zfinal.csv', sep=',', encoding='utf-8', header=False, index=False)
#    # to save to csv
#    pdcombX = pd.DataFrame( np.append( y[...,None], combX, 1) )
#    pdcombX.to_csv('datasets//combX.csv', sep=',', encoding='utf-8', header=False, index=False)

    
##############################################################################################################
## 1b) Load NME_dist_wimgfeature results
##############################################################################################################
labeltype = 'NME_dist_wimgfeature' # roilabel_NME_dist
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_wimgfeatures'
logging.basicConfig(level=logging.DEBUG)  
#logging.basicConfig(filename=os.path.join(save_to,'allDEC_exp2_y{}.log'.format(labeltype)), format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)                    

with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_dynamic.pklz'), 'rb') as fin:
    allNMEs_dynamic = pickle.load(fin)
    
with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_morphology.pklz'), 'rb') as fin:
    allNMEs_morphology = pickle.load(fin)        
    
with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_texture.pklz'), 'rb') as fin:
    allNMEs_texture = pickle.load(fin)
    
with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_stage1.pklz'), 'rb') as fin:
    allNMEs_stage1 = pickle.load(fin) 
                 
print 'Normalizing dynamic..  \n min={}, \n max={} \n'.format(np.min(allNMEs_dynamic, 0), np.max(allNMEs_dynamic, 0))
x_min, x_max = np.min(allNMEs_dynamic, 0), np.max(allNMEs_dynamic, 0)
x_max[x_max==0]=1.0e-07
normdynamic = (allNMEs_dynamic - x_min) / (x_max - x_min)
print(np.min(normdynamic, 0))
print(np.max(normdynamic, 0))

print 'Normalizing morphology..  \n min={}, \n max={} \n'.format(np.min(allNMEs_morphology, 0), np.max(allNMEs_morphology, 0))
x_min, x_max = np.min(allNMEs_morphology, 0), np.max(allNMEs_morphology, 0)
x_max[x_max==0]=1.0e-07
normorpho = (allNMEs_morphology - x_min) / (x_max - x_min)
print(np.min(normorpho, 0))
print(np.max(normorpho, 0))
    
print 'Normalizing texture..  \n min={}, \n max={} \n'.format(np.min(allNMEs_texture, 0), np.max(allNMEs_texture, 0))
x_min, x_max = np.min(allNMEs_texture, 0), np.max(allNMEs_texture, 0)
x_max[x_max==0]=1.0e-07
normtext = (allNMEs_texture - x_min) / (x_max - x_min)
print(np.min(normtext, 0))
print(np.max(normtext, 0))

print 'Normalizing stage1..  \n min={}, \n max={} \n'.format(np.min(allNMEs_stage1, 0), np.max(allNMEs_stage1, 0))
x_min, x_max = np.min(allNMEs_stage1, 0), np.max(allNMEs_stage1, 0)
x_min[np.isnan(x_min)]=1.0e-07
x_max[np.isnan(x_max)]=1.0
normstage1 = (allNMEs_stage1 - x_min) / (x_max - x_min)
normstage1[np.isnan(normstage1)]=1.0e-07
print(np.min(normstage1, 0))
print(np.max(normstage1, 0))    
                 
# shape input (798,624)
combX_allNME = np.concatenate((alldiscrSERcounts, nxGdiscfeatures, normdynamic, normorpho, normtext, normstage1), axis=1)          
YnxG_allNME = [nxGdatafeatures['roi_id'].values,
        nxGdatafeatures['roi_label'].values,
        nxGdatafeatures['roiBIRADS'].values,
        nxGdatafeatures['NME_dist'].values,
        nxGdatafeatures['NME_int_enh'].values]

print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0])   )
                 
# dfine num_centers according to clustering variable
## use y_dec to  minimizing KL divergence for clustering with known classes
X = combX_allNME  
input_size = X.shape[1]
latent_size = [input_size/rxf for rxf in [25,15,10,5]]

ysup = YnxG_allNME[1]+'_'+YnxG_allNME[3] 
ysup = ['K'+rl[1::] if rl[0]=='U' else rl for rl in ysup]
roi_labels = YnxG_allNME[1]  
roi_labels = ['K' if rl=='U' else rl for rl in roi_labels]
num_centers = len(np.unique(ysup))
fig = plt.figure(figsize=(20,6))
ax = plt.axes()
plt.gca().set_color_cycle(['red', 'green', 'blue', 'cyan'])

from decModel_wimgF import *

# to load a prevously DEC model
dfDEC_perf = pd.DataFrame()
for znum in latent_size:
    # Read autoencoder: note is not dependent on number of clusters just on z latent size
    print "... loading a prevously DEC model of latent size znum = ",znum
    dec_model = DECModel(mx.cpu(), combX_allNME, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels')
    # to load best args found during DEC
    with gzip.open(os.path.join(save_to,'dec_model_znum{}_wimgfeatures_y{}.arg'.format(znum,labeltype)), 'rb') as fu:
        best_args = pickle.load(fu)
    # algorithm to find the epoch at best acc   
    max_acc = 0
    epochs = []; best_acc = []
    for k,acc in enumerate(best_args['acci']):
        if(acc >= max_acc):
            epochs.append(k)
            best_acc.append(acc)
            max_acc = acc
    # to plot best acci
    dfDEC_perf = dfDEC_perf.append( pd.DataFrame({'bestacci': pd.Series(best_acc), 'iterations':epochs, 'Z-spaceDim':znum }) )
    ax.plot(epochs, best_acc, '.-')
    
# add legend    
plt.legend([str(latsize) for latsize in latent_size], loc='upper left')
# find max
np.max(dfDEC_perf['bestacci'])
print  dfDEC_perf[dfDEC_perf['bestacci'] == np.max(dfDEC_perf['bestacci'])]
fig.savefig(save_to+os.sep+'allDEC_wimgfeatures_bestacci_y{} Dataset_unsuprv acc vs iteration.pdf'.format(labeltype), bbox_inches='tight')    

###################################################################################################
# Calculate 5-nn TPR and TRN among pathological lesions
# Sensitivity (also called the true positive rate, the recall, or probability of detection in some fields) 
# measures the proportion of positives that are correctly identified as such (i.e. the percentage of sick people who are correctly identified as having the condition).
# Specificity (also called the true negative rate) measures the proportion of negatives that are correctly identified as such 
# create sklearn.neighbors
import sklearn.neighbors 
from utilities import visualize_Zlatent_NN_fortsne_id
import matplotlib.patches as mpatches

## 1d) Load decModel_exp2 results
labeltype = 'NME_dist_int_enh' # roilabel_NME_dist
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_exp2'
logging.basicConfig(level=logging.DEBUG) 

X = combX_allNME  
input_size = X.shape[1]
latent_size = [input_size/rxf for rxf in [25,15,10,5]]

ysup = YnxG_allNME[1]+'_'+YnxG_allNME[3]+'_'+YnxG_allNME[4] 
ysup = ['K'+rl[1::] if rl[0]=='U' else rl for rl in ysup]
roi_labels = YnxG_allNME[1]  
roi_labels = ['K' if rl=='U' else rl for rl in roi_labels]
num_centers = len(np.unique(ysup))

fig = plt.figure(figsize=(20,6))
ax = plt.axes()
plt.gca().set_color_cycle(['red', 'green', 'blue', 'cyan'])

# to load a prevously DEC model
dfDEC_perf = pd.DataFrame()
for znum in latent_size:
    # Read autoencoder: note is not dependent on number of clusters just on z latent size
    print "... loading a prevously DEC model of latent size znum = ",znum
    dec_model = DECModel(mx.cpu(), combX_allNME, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels')
    # to load best args found during DEC
    with gzip.open(os.path.join(save_to,'dec_model_znum{}_exp2_y{}.arg'.format(znum,labeltype)), 'rb') as fu:
        best_args = pickle.load(fu)
    
    # extact best params
    print(best_args['bestacci'])
    for key, v in dec_model.args.items():
        print key
        if(key=='dec_mu'):
            pass
        else:
            dec_model.args[key] = mx.nd.array(best_args[key], ctx=dec_model.xpu)

    # deal with centroids 
    dec_model.args['dec_mu'] = best_args['dec_mu']
    # extact embedding space
    all_iter = mx.io.NDArrayIter({'data': X}, batch_size=X.shape[0], shuffle=False,
                                      last_batch_handle='pad')   
    ## embedded point zi 
    zbestacci = model.extract_feature(dec_model.feature, dec_model.args, None, all_iter, X.shape[0], dec_model.xpu).values()[0]
    y = np.asarray(roi_labels)
    
    # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=200,
         init='pca', random_state=0, verbose=2, method='exact')
    Z_tsne = tsne.fit_transform(zbestacci)
    
    # plot initial z        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plot_embedding_unsuper_NMEdist_intenh(Z_tsne, best_args['named_y'], ax, title='{} tsne with perplexity {}'.format(labeltype,15), legend=True)
    
    ##########################
    # Calculate 5-nn TPR and TRN among pathological lesions
    Z_embedding_tree = sklearn.neighbors.BallTree(zbestacci, leaf_size=5)     
    # This finds the indices of 5 closest neighbors
    N = sum(y==np.unique(y)[0]) #for B
    P = sum(y==np.unique(y)[2]) #for M
    TP = []
    TN = []
    for k in range(zbestacci.shape[0]):
        iclass = y[k]
        dist, ind = Z_embedding_tree.query([zbestacci[k]], k=6)
        dist5nn, ind5nn = dist[k!=ind], ind[k!=ind]
        class5nn = y[ind5nn]
        # exlcude U class
        class5nn = class5nn[class5nn!='K']
        if(len(class5nn)>0):
            predc=[]
            for c in np.unique(class5nn):
                predc.append( sum(class5nn==c) )
            # predicion based on majority
            predclass = np.unique(class5nn)[predc==max(predc)]
            
            if(len(predclass)==1):
                # compute TP if M    
                if(iclass=='M'):
                    TP.append(predclass[0]==iclass)
                 # compute TN if B
                if(iclass=='B'):
                    TN.append(predclass[0]==iclass)
                    
            if(len(predclass)==2):
                # compute TP if M    
                if(iclass=='M'):
                    TP.append(predclass[1]==iclass)
                # compute TN if B
                if(iclass=='B'):
                    TN.append(predclass[0]==iclass)
    
    # compute TPR and TNR
    TPR = sum(TP)/float(P)
    TNR = sum(TN)/float(N)
    Accu = sum(TP+TN)/float(P+N)
    print"True Posite Rate (TPR) = %f " % TPR
    print"True Negative Rate (TNR) = %f " % TNR
    print"Accuracy (Acc) = %f " % Accu
    numc_TPR.append( TPR )               
    numc_TNR.append( TNR )
    numc_Accu.append( Accu )

    ##########################                
    # Calculate normalized MI:
    # find the relative frequency of points in Wk and Cj
    N = combX_allNME.shape[0]
    num_classes = len(np.unique(roi_labels)) # present but not needed during AE training
    classes = np.unique(roi_labels)
    roi_labels = np.asarray(roi_labels)
    # to get final cluster memberships
    pfinal = best_args['p']
    W = pfinal.argmax(axis=1)
    num_clusters = len(np.unique(W))
    clusters = np.unique(W)
    
    MLE_kj = np.zeros((num_clusters,num_classes))
    absWk = np.zeros((num_clusters))
    absCj = np.zeros((num_classes))
    for k in range(num_clusters):
        # find poinst in cluster k
        absWk[k] = sum(W==k)
        for j in range(num_classes):
            # find points of class j
            absCj[j] = sum(roi_labels==classes[j])
            # find intersection 
            ptsk = W==k 
            MLE_kj[k,j] = sum(ptsk[roi_labels==classes[j]])
    # if not assignment incluster
    absWk[absWk==0]=0.00001
    
    # compute NMI
    numIwc = np.zeros((num_clusters,num_classes))
    for k in range(num_clusters):
        for j in range(num_classes):
            if(MLE_kj[k,j]!=0):
                numIwc[k,j] = MLE_kj[k,j]/N * np.log( N*MLE_kj[k,j]/(absWk[k]*absCj[j]) )
            
    Iwk = np.sum(np.sum(numIwc, axis=1), axis=0)       
    Hc = -np.sum(absCj/N*np.log(absCj/N))
    Hw = -np.sum(absWk/N*np.log(absWk/N))
    NMI = Iwk/((Hc+Hw)/2)
    normalizedMI.append( NMI ) 
    print "... DEC normalizedMI = ", NMI