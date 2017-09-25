# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 12:09:35 2017

@author: DeepLearning
"""


from utilities import *
import sys
import os
import mxnet as mx
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
from utilities import *
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import *
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
from utilities import * 
from decModel_exp5_wimgF import *  # decModel_exp5_wimgF

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
    
#####################################
# read features     
#####################################
combX_allNME, YnxG_allNME, combX_filledbyBC, YnxG_filledbyBC = read_nxGwimg_features()
# for onlynxG features experiments
#combX_allNME, YnxG_allNME, combX_filledbyBC, YnxG_filledbyBC = read_onlynxG_features()

input_size = combX_allNME.shape[1]
latent_size =  [input_size/rxf for rxf in [25,15,10,5]]
varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(3,12,10)]
labeltype = 'roilabel_&NMEdist_wimgG' 
dec_model_load = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_exp5_wimgF'
#labeltype = 'roilabel_&NMEdist_onlynxG' 

datalabels = YnxG_allNME[1]
FUstatus = YnxG_allNME[8]
FUtime = YnxG_allNME[9]
dataXorig = combX_allNME
# organize labels, if "U" add follow up
datalabels_wFU = datalabels+['_'+str(FUs) if FUs!=None else '' for FUs in FUstatus ]
datalabels_wFUwt = datalabels+['_'+str(FUs) if FUs!=None else '' for FUs in FUstatus ]+['_'+str(FUt)  if FUt!=None else '' for FUt in FUtime ]
datalabels_wStables = datalabels+['_'+str(FUs) if FUs=='stable' else '' for FUs in FUstatus ]

#####################################
# RFmodel performance on orig X     
#####################################
# to compute RF in original
RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=1)

## calculate RFmodel ROC
# shuffle and split training and test sets
skf = StratifiedKFold(n_splits=5, shuffle=False)
skf.get_n_splits(dataXorig, datalabels_wStables)
ally_score = []
ally_test = []   
for train_index, test_index in skf.split(dataXorig, datalabels_wStables):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = dataXorig[train_index], dataXorig[test_index]
    y_train, y_test = datalabels_wStables[train_index], datalabels_wStables[test_index]
    
    # train RF model on stratitied kfold
    RFmodel = RFmodel.fit(X_train, y_train)
    y_score = RFmodel.predict_proba(X_test)
    # pass y_scores as : array, shape = [n_samples] Target scores, can either be probability estimates of the positive class..
    ally_score.append(y_score)
    ally_test.append(y_test)
    
stack_ally_score_origX = np.vstack(([ally_score[i] for i in range(len(ally_score))]))
stack_ally_test_origX = np.hstack(([ally_test[i] for i in range(len(ally_test))]))
    
# Compute ROC curve and ROC area for each class
fpr_origX = [];  tpr_origX = []; roc_aucXOriginal = {}  
classes = np.unique(y_train)
for i in range(len(classes)):
    # make malignant class = 1 (positive), Benigng = -1
    ally_test_int = [1 if l==classes[i] else -1 for l in stack_ally_test_origX]
    fpr_origX, tpr_origX, _ = roc_curve(ally_test_int, stack_ally_score_origX[:, i])
    roc_aucXOriginal[i] = auc(fpr_origX, tpr_origX)   

print 'XOriginal pooled kStra cv AUC = %f' % np.mean(roc_aucXOriginal.values()[0,1,3]) 
print roc_aucXOriginal

#####################################
# RFmodel performance on latent_size  
#####################################
scoresM = np.zeros((len(varying_mu), len(latent_size), 1))
best_roc_aucZlatent = 0.0

for ic,num_centers in enumerate(varying_mu): 
    for ik,znum in enumerate(latent_size):
        # to load a prevously DEC model
        with gzip.open(os.path.join(dec_model_load,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fin:
            dec_model = pickle.load(fin)              
        print 'dec_model initAcc = {}'.format( dec_model['initAcc'] )
        print 'dec_model bestacci = {}'.format( dec_model['bestacci'][-1] )

        ######## FIND AUC of RF in Z latent space
        datalabels = YnxG_allNME[1]
        X = combX_allNME
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
        
        # combine both latnet space dimensionality and cluster membership
        dataZspace = np.concatenate((zbestacci, pbestacci), axis=1) #zbestacci #dec_model['zbestacci']   
        RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=1)
        
        ## calculate RFmodel ROC
        # shuffle and split training and test sets
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        skf.get_n_splits(dataZspace, datalabels)
        ally_score = []
        ally_test = []   
        for train_index, test_index in skf.split(dataZspace, datalabels):
            #print("\nTRAIN:", train_index, "\nTEST:", test_index, "\n=====================\n")
            X_train, X_test = dataZspace[train_index], dataZspace[test_index]
            y_train, y_test = datalabels[train_index], datalabels[test_index]
            
            # train RF model on stratitied kfold
            RFmodel = RFmodel.fit(X_train, y_train)
            y_score = RFmodel.predict_proba(X_test)
            # pass y_scores as : array, shape = [n_samples] Target scores, can either be probability estimates of the positive class..
            ally_score.append(y_score)
            ally_test.append(y_test)
            
        stack_ally_score = np.vstack(([ally_score[i] for i in range(len(ally_score))]))
        stack_ally_test = np.hstack(([ally_test[i] for i in range(len(ally_test))]))
            
        # Compute ROC curve and ROC area for each class
        fprs = [];  tprs = []; roc_aucZlatent = {}  
        classes = np.unique(y_train)
        for i in range(len(classes)):
            # make malignant class = 1 (positive), Benigng = -1
            ally_test_int = [1 if l==classes[i] else -1 for l in stack_ally_test]
            fpr, tpr, _ = roc_curve(ally_test_int, stack_ally_score[:, i])
            roc_aucZlatent[i] = auc(fpr, tpr)   
            
        mean_roc_aucZlatent = np.mean(roc_aucZlatent.values()[0:2])
        print 'Zlatent pooled kStra cv AUC = %f' %  mean_roc_aucZlatent
        print roc_aucZlatent
        scoresM[ic,ik,0] = mean_roc_aucZlatent  #dec_model['bestacci'][-1]
        # find best ic, ik
        if( best_roc_aucZlatent < mean_roc_aucZlatent ):
            best_roc_aucZlatent = mean_roc_aucZlatent
            best_ic = ic
            best_ik = ik
            best_stack_ally_score = stack_ally_score
            best_stack_ally_test = stack_ally_test
        
## plot grid results and max performance    
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter

figscoresM, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 15)) 
colors=plt.cm.rainbow(np.linspace(0,1,6)) # plt.cm.
ax = axes.flatten()
im = ax[0].imshow(scoresM[:,:,0], cmap='viridis', interpolation='nearest')
ax[0].grid(False)
for v in range(len(varying_mu)):
    for u in range(len(latent_size)):        
        ax[0].text(u,v, '{:.2f}'.format(scoresM[v,u,0]), color='grey',
                     fontdict={'weight': 'bold', 'size': 10})
# set ticks
ax[0].xaxis.set_major_locator(FixedLocator(np.linspace(0,4,5)))
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax[0].xaxis.set_minor_locator(MultipleLocator(1))
znum_labels = [str(znum) for znum in latent_size]
ax[0].set_xticklabels(znum_labels, minor=False)
ax[0].yaxis.set_major_locator(FixedLocator(np.linspace(0,9,10)))
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax[0].yaxis.set_minor_locator(MultipleLocator(1))
mu_labels = [str(mu) for mu in varying_mu]
ax[0].set_yticklabels(mu_labels, minor=False)
ax[0].xaxis.set_label('Zspace dim')
ax[0].yaxis.set_label('# clusters')
ax[0].set_title("cvRF zDEC meanAUC")   

# plot ROC
ally_test_int = [1 if l=='B' else -1 for l in stack_ally_test_origX]
fpr_origX, tpr_origX, _ = roc_curve(ally_test_int, stack_ally_score_origX[:, 0])
ax[1].plot(fpr_origX, tpr_origX, color=colors[0], lw=2, linestyle='--', label='original HD AUC ={}'.format(auc(fpr_origX, tpr_origX) ) )
# make malignant class = 1 (positive), Benigng = -1
ally_test_int = [1 if l=='B' else -1 for l in best_stack_ally_test]
fpr, tpr, _ = roc_curve(ally_test_int, best_stack_ally_score[:, 0])
ax[1].plot(fpr, tpr, color=colors[4], lw=2, label='Zlatent AUC ={}'.format(auc(fpr, tpr) ) )
# add 50/50 line prob
ax[1].plot([0, 1], [0, 1], color='grey', lw=1, linestyle=':')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Kfold B.vs.all ROC') # 
ax[1].legend(loc="lower right")


ally_test_int = [1 if l=='M' else -1 for l in stack_ally_test_origX]
fpr_origX, tpr_origX, _ = roc_curve(ally_test_int, stack_ally_score_origX[:, 1])
ax[2].plot(fpr_origX, tpr_origX, color=colors[0], lw=2, linestyle='--', label='original HD AUC ={}'.format(auc(fpr_origX, tpr_origX) ) )
# make malignant class = 1 (positive), Benigng = -1
ally_test_int = [1 if l=='M' else -1 for l in best_stack_ally_test]
fpr, tpr, _ = roc_curve(ally_test_int, best_stack_ally_score[:, 1])
ax[2].plot(fpr, tpr, color=colors[4], lw=2, label='Zlatent AUC ={}'.format(auc(fpr, tpr) ) )
# add 50/50 line prob
ax[2].plot([0, 1], [0, 1], color='grey', lw=1, linestyle=':')
ax[2].set_xlim([0.0, 1.0])
ax[2].set_ylim([0.0, 1.05])
ax[2].set_xlabel('False Positive Rate')
ax[2].set_ylabel('True Positive Rate')
ax[2].set_title('Kfold M.vs.all ROC') # 
ax[2].legend(loc="lower right")

figscoresM.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\final-Zspace perfromace AUC ROC_{}.pdf'.format(labeltype), bbox_inches='tight')    


#####################################
# Visualizations of embedded space clusters and Follow-up status
#####################################
num_centers = 5
znum = 124
# to load a prevously DEC model
with gzip.open(os.path.join(dec_model_load,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fin:
    dec_model = pickle.load(fin)              
print 'dec_model initAcc = {}'.format( dec_model['initAcc'] )
print 'dec_model bestacci = {}'.format( dec_model['bestacci'][-1] )

######## FIND AUC of RF in Z latent space
X = combX_allNME
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

# combine both latnet space dimensionality and cluster membership
dataZspace = np.concatenate((zbestacci, pbestacci), axis=1) #zbestacci #dec_model['zbestacci'] 
y_tsne = ['FU'+rl[1::] if rl[0]=='U' else rl for rl in datalabels_wFU]
y_tsne_wFU = [FUl.split('_')[0] if len(FUl.split('_'))>1 else FUl for FUl in y_tsne] 

# For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
tsne = TSNE(n_components=2, perplexity=15, learning_rate=200,
     init='pca', random_state=0, verbose=2, method='exact')
Z_tsne = tsne.fit_transform(zbestacci)
    
# plot initial z        
from utilities import plot_embedding_unsuper_wFU
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(1,1,1)
l_clusters, Z_ind_clusters = plot_embedding_unsuper_wFU(Z_tsne, y_tsne, ax, title='lesionid_Z_ind', legend=True)
fig.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\final-t-SNE_Zspace_wFU_{}.pdf'.format(labeltype), bbox_inches='tight')    

# analize for cluster 3 FU_stables
l = l_clusters[1]
Z_ind = Z_ind_clusters[1]
y_tsne_cluster =  np.asarray(y_tsne)[l]
focus = y_tsne_cluster=='FU_stable'
focus_l =  np.asarray(l)[focus]
focusZ_ind = Z_ind[focus]
lesionid_Z_ind = YnxG_allNME[0][focus_l]

# anot
num_pts = len(focus_l)
for k in range(num_pts):
    ax.annotate(str(lesionid_Z_ind[k]), xy=(focusZ_ind[k][0], focusZ_ind[k][1]),
                xytext=(0.95-1.0*k/num_pts, 1.05),
                xycoords='data',
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->"))

fig.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\zoomed_clu3t-SNE_Zspace_wFU_{}.pdf'.format(labeltype), bbox_inches='tight')    


# analize for cluster 2 FU_stables
l = l_clusters[1]
Z_ind = Z_ind_clusters[1]
y_tsne_cluster =  np.asarray(y_tsne)[l]
focus = y_tsne_cluster=='B'
focus_l =  np.asarray(l)[focus]
focusZ_ind = Z_ind[focus]
lesionid_Z_ind = YnxG_allNME[0][focus_l]

# anot
num_pts = len(focus_l)
for k in range(num_pts):
    ax.annotate(str(lesionid_Z_ind[k]), xy=(focusZ_ind[k][0], focusZ_ind[k][1]),
                xytext=(0.95-1.0*k/num_pts, 1.05),
                xycoords='data',
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->"))

fig.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\zoomed_clu2Benign-SNE_Zspace_wFU_{}.pdf'.format(labeltype), bbox_inches='tight')    


# analize for cluster 2 FU_stables
l = l_clusters[3]
Z_ind = Z_ind_clusters[3]
y_tsne_cluster =  np.asarray(y_tsne)[l]
focus = y_tsne_cluster=='M'
focus_l =  np.asarray(l)[focus]
focusZ_ind = Z_ind[focus]
lesionid_Z_ind = YnxG_allNME[0][focus_l]

# anot
num_pts = len(focus_l)
for k in range(num_pts):
    ax.annotate(str(lesionid_Z_ind[k]), xy=(focusZ_ind[k][0], focusZ_ind[k][1]),
                xytext=(0.95-1.0*k/num_pts, 1.05),
                xycoords='data',
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->"))

fig.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\zoomed_clu1Malignant-SNE_Zspace_wFU_{}.pdf'.format(labeltype), bbox_inches='tight')    


# analize for cluster 0 FU_stables
y_tsne = ['FU'+rl[1::] if rl[0]=='U' else rl for rl in datalabels_wFUwt]
y_tsne_wFU = [FUl.split('_')[0] if len(FUl.split('_'))>1 else FUl for FUl in y_tsne] 

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(1,1,1)
l_clusters, Z_ind_clusters = plot_embedding_unsuper_wFU(Z_tsne, y_tsne, ax, title='{} tsne with perplexity {}'.format(labeltype,15), legend=True)
fig.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\final-t-SNE_Zspace_wFUwtimeframe_{}.pdf'.format(labeltype), bbox_inches='tight')    


l = l_clusters[1]
Z_ind = Z_ind_clusters[1]
y_tsne_cluster =  np.asarray(y_tsne)[l]
focus = y_tsne_cluster=='FU_stable_2years'
focus_l =  np.asarray(l)[focus]
focusZ_ind = Z_ind[focus]
lesionid_Z_ind = YnxG_allNME[0][focus_l]

# anot
num_pts = len(focus_l)
for k in range(num_pts):
    ax.annotate(str(lesionid_Z_ind[k]), xy=(focusZ_ind[k][0], focusZ_ind[k][1]),
                xytext=(0.95-1.0*k/num_pts, 1.05),
                xycoords='data',
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->"))

fig.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\zoomed_clu0FU_Stable_2y-SNE_Zspace_wFU_{}.pdf'.format(labeltype), bbox_inches='tight')    











