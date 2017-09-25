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
import data
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import mode15l
from autoencoder import AutoEncoderModel
from solver import Solver, Monitor
import logging

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
combX_allNME, YnxG_allNME, combX_filledbyBC, YnxG_filledbyBC = read_nxGwimg_features()
# for onlynxG features experiments
#combX_allNME, YnxG_allNME, combX_filledbyBC, YnxG_filledbyBC = read_onlynxG_features()

input_size = combX_allNME.shape[1]
latent_size =  [input_size/rxf for rxf in [25,15,10,5]]
varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(3,8,6)]
labeltype = 'roilabel_&NMEdist_wimgG' 
dec_model_load = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_exp5_wimgF_updated'
#dec_model_load = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_exp5'
#labeltype = 'roilabel_&NMEdist_onlynxG' 

cvRFZspaceAccuracy = []
cvRFZspaceNME_DISTAcc = []
cvRFZspaceNME_INTENHAcc = []
mean_5cvRFZspaceNME = []

cvRFOriginalXAccuracy = []
cvRFNME_DISTAcc = []
cvRFNME_INTENHAcc = []
mean_5cvRFNMEdes = []

scoresM = np.zeros((len(varying_mu), len(latent_size), 8))
scoresM_titles=[]
for ic,num_centers in enumerate(varying_mu): 
    for ik,znum in enumerate(latent_size):
        # to load a prevously DEC model
        with gzip.open(os.path.join(dec_model_load,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'rb') as fin:
            dec_model = pickle.load(fin)              
        print 'dec_model initAcc = {}'.format( dec_model['initAcc'] )
        print 'dec_model bestacci = {}'.format( dec_model['bestacci'][-1] )
#        scoresM[ic,ik,0] = dec_model['bestacci'][-1]
#        scoresM_titles.append("DEC bestacci")
        
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
            print("\nTRAIN:", train_index, "\nTEST:", test_index, "\n=====================\n")
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
        figsROC, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7)) 
        classes = np.unique(y_train)
        colors=plt.cm.gist_rainbow(np.linspace(0,1,len(classes))) # plt.cm.
        for i in range(len(classes)):
            # make malignant class = 1 (positive), Benigng = -1
            ally_test_int = [1 if l==classes[i] else -1 for l in stack_ally_test]
            fpr, tpr, _ = roc_curve(ally_test_int, stack_ally_score[:, i])
            roc_aucZlatent[i] = auc(fpr, tpr)   
            # plot
            axes[0].plot(fpr, tpr, color=colors[i], lw=2, label='AUC {} ={}'.format(classes[i],roc_aucZlatent[i]) )
        
        print 'Zlatent pooled kStra cv AUC = %f' % np.mean(roc_aucZlatent.values()) 
        print roc_aucZlatent
        scoresM[ic,ik,0] = np.mean(roc_aucZlatent.values())  #dec_model['bestacci'][-1]
        scoresM_titles.append("DEC bestAUC")
        
        axes[0].plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('Zlatent: StratKFold pooled ROC validation')
        axes[0].legend(loc="lower right")

        # other model metrics
        print 'cvRF_nme_dist= {}'.format( dec_model['cvRF_nme_dist'] )
        print 'cvRF_nme_intenh = {}'.format( dec_model['cvRF_nme_intenh'])
        RF_5cv_nmeDes = np.mean([dec_model['cvRF_nme_dist'],dec_model['cvRF_nme_intenh']])
        print 'mean RF_5cv_nmeDes = {}'.format( RF_5cv_nmeDes )
        scoresM[ic,ik,1] = dec_model['cvRF_nme_dist']
        scoresM[ic,ik,2] = dec_model['cvRF_nme_intenh']
        scoresM[ic,ik,3] = RF_5cv_nmeDes
        scoresM_titles.append("cvRF_nme_dist")
        scoresM_titles.append("cvRF_nme_intenh")
        scoresM_titles.append("mean_RF_5cv_nmeDes")
        
        ######## train an RF on ORIGINAL space Finally perform a cross-validation using RF
        datalabels = YnxG_allNME[1]
        dataXorig = combX_allNME
        RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=1)
        
        ## calculate RFmodel ROC
        # shuffle and split training and test sets
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        skf.get_n_splits(dataXorig, datalabels)
        ally_score = []
        ally_test = []   
        for train_index, test_index in skf.split(dataXorig, datalabels):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = dataXorig[train_index], dataXorig[test_index]
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
        fprs = [];  tprs = []; roc_aucXOriginal = {}  
        classes = np.unique(y_train)
        colors=plt.cm.gist_rainbow(np.linspace(0,1,len(classes))) # plt.cm.
        for i in range(len(classes)):
            # make malignant class = 1 (positive), Benigng = -1
            ally_test_int = [1 if l==classes[i] else -1 for l in stack_ally_test]
            fpr, tpr, _ = roc_curve(ally_test_int, stack_ally_score[:, i])
            roc_aucXOriginal[i] = auc(fpr, tpr)   
            # plot
            axes[1].plot(fpr, tpr, color=colors[i], lw=2, label='AUC {} ={}'.format(classes[i],roc_aucXOriginal[i]) )
        
        print 'XOriginal pooled kStra cv AUC = %f' % np.mean(roc_aucXOriginal.values()) 
        print roc_aucXOriginal
        scoresM[ic,ik,4] = np.mean(roc_aucXOriginal.values())  #dec_model['bestacci'][-1]
        scoresM_titles.append("OriginalX bestAUC")
        
        axes[1].plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('XOriginal: StratKFold pooled ROC validation')
        axes[1].legend(loc="lower right")
        figsROC.suptitle('pooled_AUC_DEC = {}, znum={}, num_centers={}'.format(round(np.mean(roc_aucZlatent.values()),3),znum,num_centers))
        figsROC.savefig(dec_model_load+os.sep+'kStra_pooled_AUC_Zspace_znum{}_numc{}_{}.pdf'.format(znum,num_centers,labeltype), bbox_inches='tight')    
        
        # Evaluate a score by cross-validation
        # integer=5, to specify the number of folds in a (Stratified)KFold,
#        scores = cross_val_score(RFmodel, data, datalabels, cv=5)
#        print "(mean 5cv cv OriginalX RF_Accuracy = %f " % scores.mean()      
#        print scores.tolist()
#        scoresM[ic,ik,4] = scores.mean()
#        scoresM_titles.append("OriginalX RF_Accuracy ")
        
        ####### train an RF on ORIGINAL space Finally perform a cross-validation using RF   
        datalabels = YnxG_filledbyBC[3]
        data = combX_filledbyBC
        RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=1)
        # Evaluate a score by cross-validation
        # integer=5, to specify the number of folds in a (Stratified)KFold,
        scores_nme_dist = cross_val_score(RFmodel, data, datalabels, cv=5)
        print "(cvRF OriginalX_nme_distAcc = %f " % scores_nme_dist.mean()      
        print scores_nme_dist.tolist()
        scoresM[ic,ik,5] = scores_nme_dist.mean()
        scoresM_titles.append("OriginalX_nme_distAcc")
        
        ######## train an RF on ORIGINAL space Finally perform a cross-validation using RF   
        datalabels = YnxG_filledbyBC[4]
        data = combX_filledbyBC
        RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=1)
        # Evaluate a score by cross-validation
        # integer=5, to specify the number of folds in a (Stratified)KFold,
        scores_nme_intenh = cross_val_score(RFmodel, data, datalabels, cv=5)
        print "(cvRF OriginalX_nme_intenhcc = %f " % scores_nme_intenh.mean()      
        print scores_nme_intenh.tolist()
        scoresM[ic,ik,6] = scores_nme_intenh.mean()
        scoresM_titles.append("OriginalX_nme_intenhcc")
        
        RF_5cv_OriginalX_nmeDes = np.mean([ scores_nme_dist.mean(),scores_nme_intenh.mean() ])
        print 'mean RF_5cv_OriginalX_nmeDes = {}'.format( RF_5cv_OriginalX_nmeDes )
        scoresM[ic,ik,7] = RF_5cv_OriginalX_nmeDes
        scoresM_titles.append("mean RF_5cv_OriginalX_nmeDes")
        
        

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter

figscoresM, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 15)) 
for k,ax in enumerate(axes.flat):
    im = ax.imshow(scoresM[:,:,k], cmap='viridis', interpolation='nearest')
    ax.grid(False)
    for v in range(len(varying_mu)):
        for u in range(len(latent_size)):        
            ax.text(u,v, '{:.2f}'.format(scoresM[v,u,k]), color=np.array([0.65,0.65,0.65,1]),
                         fontdict={'weight': 'bold', 'size': 8})
    # set ticks
    ax.xaxis.set_major_locator(FixedLocator(np.linspace(0,4,5)))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    znum_labels = [str(znum) for znum in latent_size]
    ax.set_xticklabels(znum_labels, minor=False)
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0,6,7)))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    mu_labels = [str(mu) for mu in varying_mu]
    ax.set_yticklabels(mu_labels, minor=False)
    ax.xaxis.set_label('Zspace dim')
    ax.yaxis.set_label('# clusters')
    ax.set_title(scoresM_titles[k])
   
figscoresM.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\scoresM_AUC_{}_final.pdf'.format(labeltype), bbox_inches='tight')    

## plot line plots
dict_aucZlatent = pd.DataFrame() 
for k,znum in enumerate(latent_size):
    for l,num_c in enumerate(varying_mu):
        dict_aucZlatent = dict_aucZlatent.append( pd.Series({'Zspacedim':znum, 'Zspace_AUC_ROC': scoresM[l,k,0], 'num_clusters':num_c}), ignore_index=True)
    
fig2 = plt.figure(figsize=(20,6))
ax2 = plt.axes()
sns.set_context("notebook")
sns.set_style("darkgrid", {"axes.facecolor": ".9"})    
sns.pointplot(x="num_clusters", y="Zspace_AUC_ROC", hue="Zspacedim", data=dict_aucZlatent, ax=ax2, size=0.05) 
ax2.xaxis.set_label('# clusters')
ax2.yaxis.set_label('Zspace AUC ROC')
ax2.set_title('Zspace AUC ROC vs. number of clusters')
fig2.savefig('Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels\\Zspace AUC ROC_{}.pdf'.format(labeltype), bbox_inches='tight')    
plt.close(fig2)

# add colorbar
#mins=[]; maxs=[]
## find min and max for scaling
#for k in range(8):
#    mins.append( np.min(scoresM[:,:,k]) )
#    maxs.append( np.max(scoresM[:,:,k]) )
#minsi = np.min(mins)
#maxsi = np.max(maxs)
#figscoresM.subplots_adjust(right=0.8)
#cbar_ax = figscoresM.add_axes([0.85, 0.15, 0.05, 0.7])
#figscoresM.colorbar(im, cax=cbar_ax)



########################################
## Final set of parse_results
########################################
# read features     
combX_allNME, YnxG_allNME, combX_filledbyBC, YnxG_filledbyBC = read_nxGwimg_features()
# for onlynxG features experiments
#combX_allNME, YnxG_allNME, combX_filledbyBC, YnxG_filledbyBC = read_onlynxG_features()

input_size = combX_allNME.shape[1]
latent_size =  [input_size/rxf for rxf in [25,15,10,5]]
varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(3,12,10)]
labeltype = 'roilabel_&NMEdist_wimgG' 
dec_model_load = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_exp5_wimgF'

datalabels = YnxG_allNME[1]
dataXorig = combX_allNME
RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=1)

## calculate RFmodel ROC
# shuffle and split training and test sets
skf = StratifiedKFold(n_splits=5, shuffle=False)
skf.get_n_splits(dataXorig, datalabels)
ally_score = []
ally_test = []   
for train_index, test_index in skf.split(dataXorig, datalabels):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = dataXorig[train_index], dataXorig[test_index]
    y_train, y_test = datalabels[train_index], datalabels[test_index]
    
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

print 'XOriginal pooled kStra cv AUC = %f' % np.mean(roc_aucXOriginal.values()[0:2]) 
print roc_aucXOriginal
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

















