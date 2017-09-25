# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:12:00 2017

@author: DeepLearning
"""

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

from sklearn.manifold import TSNE
from utilities import *
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_embedding_wcentroids(Z_tsne, y_tsne, allclusters_centroids_indx, centroids_imgs, plot_args, title=None, legend=True, withClustersImg=True, withClustersMembership=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import offsetbox
    from matplotlib.offsetbox import TextArea, AnnotationBbox
    from matplotlib.offsetbox import DrawingArea, OffsetImage

    fig = plt.figure(figsize=(20, 15))
    ax = plt.axes(frameon=False)

    x_min, x_max = np.min(Z_tsne, 0), np.max(Z_tsne, 0)
    Z_tsne = (Z_tsne - x_min) / (x_max - x_min)

    # process labels 
    classes = [str(c) for c in np.unique(y_tsne)]
    colors=plt.cm.viridis(np.linspace(0,1,len(classes))) # plt.cm.gist_rainbow
    c_patchs = []
    greyc_U = np.array([0.5,0.5,0.5,1])
    for k in range(len(classes)):
        if(classes[k]=='u'):
            c_patchs.append(mpatches.Patch(color=greyc_U, label=classes[k]))
        else:
            c_patchs.append(mpatches.Patch(color=colors[k], label=classes[k]))
        
    if(legend):
        plt.legend(handles=c_patchs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':8})
    
    if(withClustersMembership):
        clusterPred = plot_args['clusterPred']
        clustercolors=plt.cm.gist_rainbow(np.linspace(0,1,len(np.unique(clusterPred)))) # plt.cm.gist_rainbow
        for i in range(Z_tsne.shape[0]):
            kcl = clusterPred[i]
            colori = clustercolors[kcl]                               
            plt.text(Z_tsne[i, 0], Z_tsne[i, 1], '.', color=colori, fontdict={'weight': 'bold', 'size': 36})     
    else:
        for i in range(Z_tsne.shape[0]):
            for k in range(len(classes)):
                if str(y_tsne[i])=='u':
                    colori = greyc_U
                else:
                    if(str(y_tsne[i])==classes[k]):
                        colori = colors[k]                               
                        plt.text(Z_tsne[i, 0], Z_tsne[i, 1], '.', color=colori, fontdict={'weight': 'bold', 'size': 36})
    
    if(withClustersImg):
        # plot closets image to cluster centroid: one per class
        num_clusters = len(allclusters_centroids_indx)
        for j in xrange(num_clusters):
            Zind = allclusters_centroids_indx[j]
            # plot            
            imgbox = OffsetImage(centroids_imgs[j], zoom=0.25)
            ab = AnnotationBbox(imgbox, (Z_tsne[Zind,:][0], Z_tsne[Zind,:][1]),
                xybox=(0.95-1.0*j/num_clusters, 0.90),
                xycoords='data',
                boxcoords=("axes fraction", "axes fraction"),
                pad=0.0,
                frameon=False,
                arrowprops=dict(arrowstyle="->"))
            ax.add_artist(ab)   
            labels_centroidsc = y_tsne[allclusters_centroids_indx[j]].split('_')
            boxlabel = 'cluster'+str(j)+'_'+plot_args['label_centroids'][j]+'_'+y_tsne[allclusters_centroids_indx[j]]
            ax.annotate(boxlabel, xy=(0.95-1.0*j/num_clusters, 1.1), xycoords='data', xytext=(0.9-1.0*j/num_clusters, 1.1))
    
    #plt.setp(ax, xticks=(), yticks=())
    #ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.4)
    if title is not None:
        plt.title(title)
        
  
from utilities import read_nxGwimg_features, read_onlynxG_features
from utilities import make_graph_ndMRIdata
from decModel_exp5_wimgF import *     
 
## 1) read in the datasets both all NME (to do pretraining)
#####################################
# read features     
combX_allNME, YnxG_allNME, combX_filledbyBC, YnxG_filledbyBC = read_nxGwimg_features()
# for onlynxG features experiments
#combX_allNME, YnxG_allNME, combX_filledbyBC, YnxG_filledbyBC = read_onlynxG_features()
 
######################
## 2) Define Zlatent space and clusteres
######################                 
X = combX_allNME  
# do for the rest of descriptors
roi_labels = YnxG_allNME[1]  
roi_labels = np.asarray(['K' if rl=='U' else rl for rl in roi_labels], dtype=object)
YnxG_allNME[3][range(combX_filledbyBC.shape[0])] = YnxG_filledbyBC[3]
YnxG_allNME[4][range(combX_filledbyBC.shape[0])] = YnxG_filledbyBC[4]
# define labels to color code points in visuzliaztion
labels_tsne = roi_labels+'_'+YnxG_allNME[3]+'_'+YnxG_allNME[4]                           
input_size = X.shape[1]
latent_size = [input_size/rxf for rxf in [25,15,10,5]]

labeltype = 'roilabel_&NMEdist_wimgG' 
save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_exp5_wimgF'
num_centers = 5 # best results on decModel_exp5_wimgF
znum = latent_size[3]

# to load a prevously DEC model
#for znum in latent_size:
   
# to load a prevously DEC model
with gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fin:
    dec_model = pickle.load(fin)              
print 'dec_model initAcc = {}'.format( dec_model['initAcc'] )
print 'dec_model bestacci = {}'.format( dec_model['bestacci'][-1] )
  
 #####################
# Calculate normalized MI: find the relative frequency of points in Wk and Cj
#####################
N = X.shape[0]    
# extact embedding space
all_iter = mx.io.NDArrayIter({'data': X}, batch_size=X.shape[0], shuffle=False,
                                          last_batch_handle='pad')   
## embedded point zi 
aDEC = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels')
# isolate DEC model keys
dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
    'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
    'encoder_3_bias', 'encoder_2_bias']
dec_args = {key: v for key, v in dec_model.items() if key in dec_args_keys}
dec_args['dec_mubestacci'] = dec_model['dec_mubestacci']    
# convert to list of NDArrays
mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
zbestacci = model.extract_feature(aDEC.feature, mxdec_args, None, all_iter, X.shape[0], aDEC.xpu).values()[0]      
# orig paper 256*40 (10240) point for upgrade about 1/6 (N) of data
#zbestacci = dec_model['zbestacci'] 
pbestacci = np.zeros((zbestacci.shape[0], dec_model['num_centers']))
aDEC.dec_op.forward([zbestacci, dec_args['dec_mubestacci'].asnumpy()], [pbestacci])

## 1) Predict BIRADs by clustering
#####################
clusterPred = pbestacci.argmax(axis=1)
#Acc_BIRADSpred = cluster_acc(BIRADSpred, y_dec)

## PLOT labels and cluster memberships      
# find cluster centroids    
Z_embedding_tree = sklearn.neighbors.BallTree(zbestacci, leaf_size=6) 
allclusters_centroids_indx = []
BIRADS_centroids = []
label_centroids = []
centroids_imgsDEL = []
centroids_imgsMST = []
for kcl in range(dec_args['dec_mubestacci'].asnumpy().shape[0]):
    dist, ind = Z_embedding_tree.query([dec_args['dec_mubestacci'].asnumpy()[kcl]], k=1)
    print "closest neighbor to cluster %i centroid = %i" % (kcl,ind)
    allclusters_centroids_indx.append(ind[0][0])
    roi_id = YnxG_allNME[0][ind[0][0]]
    BIRADS_centroids.append( YnxG_allNME[2][ind[0][0]] )
    label_centroids.append( YnxG_allNME[1][ind[0][0]] )
    centroids_imgsDEL.append( make_graph_ndMRIdata(roi_id, 'DEL') )
    centroids_imgsMST.append( make_graph_ndMRIdata(roi_id, 'MST') )

# plot with tSNE
tsne = TSNE(n_components=2, perplexity=15, learning_rate=125,
                            init='pca', random_state=0, verbose=2, method='exact')
Z_tsne = tsne.fit_transform(zbestacci)  
    
# plot 
plot_args = {'BIRADS_centroids':BIRADS_centroids,
             'label_centroids':label_centroids,
             'clusterPred':clusterPred}
             
plot_embedding_wcentroids(Z_tsne, labels_tsne, allclusters_centroids_indx, centroids_imgsDEL, plot_args, title='t-SNE: BIRADS', legend=True, withClustersImg=True)
fig = plt.gcf()
fig.savefig(save_to+os.sep+'Z_tsne_wcentroids_{}_final.pdf'.format(labeltype), bbox_inches='tight')    

plot_embedding_wcentroids(Z_tsne, labels_tsne, allclusters_centroids_indx, centroids_imgsMST, plot_args, title='t-SNE: cluster membership', legend=False, withClustersImg=True, withClustersMembership=True)
fig = plt.gcf()
fig.savefig(save_to+os.sep+'Z_tsne_wcentroids_withClustersMembership_{}_final.pdf'.format(labeltype), bbox_inches='tight')    


## 1) Predict DESCRIPTORS by 5nns
#####################
allpred_BIRADS = []
allpred_nme_dist = []
allpred_nme_intenh = []
allpred_dce_init = []
allpred_dce_delay = []
allpred_dce_curve_type = []
for k in range(zbestacci.shape[0]):
    iclass = YnxG_allNME[2][k]+'_'+YnxG_allNME[3][k]+'_'+YnxG_allNME[4][k]+'_'+YnxG_allNME[5][k]+'_'+YnxG_allNME[6][k]+'_'+YnxG_allNME[7][k]
    dist, ind = Z_embedding_tree.query([zbestacci[k]], k=2)
    # find 5 nearest neighbors
    dist5nn, ind5nn = dist[k!=ind], ind[k!=ind]
    class5nn_BIRADS = YnxG_allNME[2][ind5nn]
    class5nn_nme_dist = YnxG_allNME[3][ind5nn]
    class5nn_nme_intenh = YnxG_allNME[4][ind5nn]
    class5nn_dce_init = YnxG_allNME[5][ind5nn]
    class5nn_dce_delay = YnxG_allNME[6][ind5nn]
    class5nn_curve_type = YnxG_allNME[7][ind5nn]
    # compute predBIRADS ACC based on nme similar local neighborhood   
    classes = np.unique(YnxG_allNME[2])                 
    predBIRADS=[]
    predBIRADS.append( sum([dist5nn[l]*int(lab=='0') for l,lab in enumerate(class5nn_BIRADS)]) )
    predBIRADS.append( sum([dist5nn[l]*int(lab=='1') for l,lab in enumerate(class5nn_BIRADS)]) )
    predBIRADS.append( sum([dist5nn[l]*int(lab=='2') for l,lab in enumerate(class5nn_BIRADS)]) )
    predBIRADS.append( sum([dist5nn[l]*int(lab=='3') for l,lab in enumerate(class5nn_BIRADS)]) )
    predBIRADS.append( sum([dist5nn[l]*int(lab=='4') for l,lab in enumerate(class5nn_BIRADS)]) )
    predBIRADS.append( sum([dist5nn[l]*int(lab=='5') for l,lab in enumerate(class5nn_BIRADS)]) )
    predBIRADS.append( sum([dist5nn[l]*int(lab=='6') for l,lab in enumerate(class5nn_BIRADS)]) )
    pred_BIRADS = [int(classes[l][0]) for l,pc in enumerate(predBIRADS) if pc>=max(predBIRADS) and max(predBIRADS)>0][0]
    allpred_BIRADS.append(pred_BIRADS)

    # final NME descriptor 
    nme_dist = np.unique(YnxG_allNME[3])
    prednmed=[]
    prednmed.append( sum([dist5nn[l]*int(lab=='Diffuse') for l,lab in enumerate(class5nn_nme_dist)]) ) 
    prednmed.append( sum([dist5nn[l]*int(lab=='Focal') for l,lab in enumerate(class5nn_nme_dist)]) )         
    prednmed.append( sum([dist5nn[l]*int(lab=='Linear') for l,lab in enumerate(class5nn_nme_dist)]) ) 
    prednmed.append( sum([dist5nn[l]*int(lab=='MultipleRegions') for l,lab in enumerate(class5nn_nme_dist)]) ) 
    prednmed.append( sum([dist5nn[l]*int(lab=='N/A') for l,lab in enumerate(class5nn_nme_dist)]) ) 
    prednmed.append( sum([dist5nn[l]*int(lab=='Regional') for l,lab in enumerate(class5nn_nme_dist)]) ) 
    prednmed.append( sum([dist5nn[l]*int(lab=='Segmental') for l,lab in enumerate(class5nn_nme_dist)]) ) 
    # predicion based on majority voting
    pred_nme_dist = [nme_dist[l] for l,pc in enumerate(prednmed) if pc>=max(prednmed) and max(prednmed)>0][0] 
    allpred_nme_dist.append(pred_nme_dist)

    # final NME descriptor 
    nme_intenh = np.unique(YnxG_allNME[4])      
    prednmeie=[]
    prednmeie.append( sum([dist5nn[l]*int(lab=='Clumped') for l,lab in enumerate(class5nn_nme_intenh)]) )
    prednmeie.append( sum([dist5nn[l]*int(lab=='ClusteredRing') for l,lab in enumerate(class5nn_nme_intenh)]) )
    prednmeie.append( sum([dist5nn[l]*int(lab=='Heterogeneous') for l,lab in enumerate(class5nn_nme_intenh)]) )
    prednmeie.append( sum([dist5nn[l]*int(lab=='Homogeneous') for l,lab in enumerate(class5nn_nme_intenh)]) )
    prednmeie.append( sum([dist5nn[l]*int(lab=='N/A') for l,lab in enumerate(class5nn_nme_intenh)]) )
    prednmeie.append( sum([dist5nn[l]*int(lab=='Stippled or punctate') for l,lab in enumerate(class5nn_nme_intenh)]) )
    # predicion based on majority voting
    pred_nme_intenh = [nme_intenh[l] for l,pc in enumerate(prednmeie) if pc>=max(prednmeie) and max(prednmeie)>0][0]
    allpred_nme_intenh.append(pred_nme_intenh)
 
    # final NME descriptor 
    dce_init = np.unique(YnxG_allNME[5]) 
    predce_init=[]
    predce_init.append( sum([dist5nn[l]*int(lab=='Medium') for l,lab in enumerate(class5nn_dce_init)])  )
    predce_init.append( sum([dist5nn[l]*int(lab=='Moderate to marked') for l,lab in enumerate(class5nn_dce_init)])  )
    predce_init.append( sum([dist5nn[l]*int(lab=='N/A') for l,lab in enumerate(class5nn_dce_init)])  )
    predce_init.append( sum([dist5nn[l]*int(lab=='Rapid') for l,lab in enumerate(class5nn_dce_init)])  )
    predce_init.append( sum([dist5nn[l]*int(lab=='Slow') for l,lab in enumerate(class5nn_dce_init)])  )
    # predicion based on majority voting
    pred_dce_init = [dce_init[l] for l,pc in enumerate(predce_init) if pc>=max(predce_init) and max(predce_init)>0][0]
    allpred_dce_init.append(pred_dce_init)
            
    # final NME descriptor 
    dce_delay = np.unique(YnxG_allNME[6]) 
    predce_delay=[]
    predce_delay.append( sum([dist5nn[l]*int(lab=='N/A') for l,lab in enumerate(class5nn_dce_delay)])  )
    predce_delay.append( sum([dist5nn[l]*int(lab=='Persistent') for l,lab in enumerate(class5nn_dce_delay)])  )
    predce_delay.append( sum([dist5nn[l]*int(lab=='Plateau') for l,lab in enumerate(class5nn_dce_delay)])  )
    predce_delay.append( sum([dist5nn[l]*int(lab=='Washout') for l,lab in enumerate(class5nn_dce_delay)])  )
    # predicion based on majority voting
    pred_dce_delay = [dce_delay[l] for l,pc in enumerate(predce_delay) if pc>=max(predce_delay) and max(predce_delay)>0][0]
    allpred_dce_delay.append(pred_dce_delay)
    
    # final NME descriptor 
    curve_type = np.unique(YnxG_allNME[7]) 
    predce_curve_type=[]
    predce_curve_type.append( sum([dist5nn[l]*int(lab=='II') for l,lab in enumerate(class5nn_curve_type)])  )
    predce_curve_type.append( sum([dist5nn[l]*int(lab=='III') for l,lab in enumerate(class5nn_curve_type)])  )
    predce_curve_type.append( sum([dist5nn[l]*int(lab=='Ia') for l,lab in enumerate(class5nn_curve_type)])  )
    predce_curve_type.append( sum([dist5nn[l]*int(lab=='Ib') for l,lab in enumerate(class5nn_curve_type)])  )
    predce_curve_type.append( sum([dist5nn[l]*int(lab=='N/A') for l,lab in enumerate(class5nn_curve_type)])  )
    predce_curve_type.append( sum([dist5nn[l]*int(lab=='Other') for l,lab in enumerate(class5nn_curve_type)])  )
    # predicion based on majority voting
    pred_dce_curve_type = [curve_type[l] for l,pc in enumerate(predce_curve_type) if pc>=max(predce_curve_type) and max(predce_curve_type)>0][0]
    allpred_dce_curve_type.append(pred_dce_curve_type)
    
    print "\ngt labels:\t %s" % iclass
    print "predicted:\t %s" % pred_BIRADS+'_'+pred_nme_dist+'_'+pred_nme_intenh+'_'+pred_dce_init+'_'+pred_dce_delay+'_'+pred_dce_curve_type

# save to R and csv
a = np.append( np.asarray(roi_labels)[...,None], np.asarray(allpred_BIRADS)[...,None], 1)
a = np.append( a, np.asarray(allpred_nme_dist)[...,None],  1)
a = np.append( a, np.asarray(allpred_nme_intenh)[...,None],  1)
a = np.append( a, np.asarray(allpred_dce_init)[...,None],  1)
a = np.append( a, np.asarray(allpred_dce_delay)[...,None],  1)
a = np.append( a, np.asarray(allpred_dce_curve_type)[...,None],  1)
a = np.append( a, zbestacci,  1)
a = np.append( a, pbestacci,  1)
pdzfinal = pd.DataFrame(a, columns=['label', 'pred_BIRADS','red_nme_dist','pred_nme_intenh','pred_dce_init','pred_dce_delay', 'pred_curve_type']+['z'+str(l+1) for l in range(znum+pbestacci.shape[1])])                                               
pdzfinal.to_csv(os.path.join(save_to,'zfinal{}_{}nn_{}.csv'.format(znum,str(1),labeltype)), sep=',', encoding='utf-8', header=False, index=False)

 