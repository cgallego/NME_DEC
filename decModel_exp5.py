# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:49:58 2017

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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp

def cluster_acc(Y_pred, Y):
    # Y_pred=ysup_pred; Y=y_dec
    # For all algorithms we set the
    # number of clusters to the number of ground-truth categories
    # and evaluate performance with unsupervised clustering ac-curacy (ACC):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        # rows are predictions, columns are ground truth
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w
    

class DECModel(model.MXModel):
    class DECLoss(mx.operator.NumpyOp):
        def __init__(self, num_centers, alpha):
            super(DECModel.DECLoss, self).__init__(need_top_grad=False)
            self.num_centers = num_centers
            self.alpha = alpha

        def forward(self, in_data, out_data):
            z = in_data[0]
            mu = in_data[1]
            q = out_data[0]
            ## eq. 1 use the Students t-distribution as a kernel to measure the similarity between embedded point zi and centroid mu j
            self.mask = 1.0/(1.0+cdist(z, mu)**2/self.alpha)
            q[:] = self.mask**((self.alpha+1.0)/2.0)
            q[:] = (q.T/q.sum(axis=1)).T
            

        def backward(self, out_grad, in_data, out_data, in_grad):
            q = out_data[0]
            z = in_data[0]
            mu = in_data[1]
            p = in_data[2]
            dz = in_grad[0]
            dmu = in_grad[1]
            self.mask *= (self.alpha+1.0)/self.alpha*(p-q) #
            # The gradients of L with respect to feature space embedding of each data point zi and each cluster centroid mu j are computed as:
            dz[:] = (z.T*self.mask.sum(axis=1)).T - self.mask.dot(mu) # eq. 4
            dmu[:] = (mu.T*self.mask.sum(axis=0)).T - self.mask.T.dot(z) # eq.5

        def infer_shape(self, in_shape):
            assert len(in_shape) == 3
            assert len(in_shape[0]) == 2
            input_shape = in_shape[0]
            label_shape = (input_shape[0], self.num_centers)
            mu_shape = (self.num_centers, input_shape[1])
            out_shape = (input_shape[0], self.num_centers)
            return [input_shape, mu_shape, label_shape], [out_shape]

        def list_arguments(self):
            return ['data', 'mu', 'label']

    def setup(self, X, num_centers, alpha, znum, save_to='dec_model'):
        # Read previously trained _SAE
        ae_model = AutoEncoderModel(self.xpu, [X.shape[1],500,500,2000,znum], pt_dropout=0.2)
        ae_model.load( os.path.join(save_to,'SAE_zsize{}.arg'.format(str(znum))) ) #_Nbatch_wimgfeatures
        logging.log(logging.INFO, "Reading Autoencoder from file..: %s"%(os.path.join(save_to,'SAE_zsize{}.arg'.format(znum))) )
        self.ae_model = ae_model
        logging.log(logging.INFO, "finished reading Autoencoder from file..: ")

        self.dec_op = DECModel.DECLoss(num_centers, alpha)
        label = mx.sym.Variable('label')
        self.feature = self.ae_model.encoder
        self.loss = self.dec_op(data=self.ae_model.encoder, label=label, name='dec')
        self.args.update({k:v for k,v in self.ae_model.args.items() if k in self.ae_model.encoder.list_arguments()})
        self.args['dec_mu'] = mx.nd.empty((num_centers, self.ae_model.dims[-1]), ctx=self.xpu)
        self.args_grad.update({k: mx.nd.empty(v.shape, ctx=self.xpu) for k,v in self.args.items()})
        self.args_mult.update({k: k.endswith('bias') and 2.0 or 1.0 for k in self.args})
        self.num_centers = num_centers
        self.best_args = {}
        self.best_args['num_centers'] = num_centers
        self.best_args['znum'] = znum

    def cluster(self, X_train, y_dec_train, y_train, classes, batch_size, save_to, labeltype, update_interval=None):
        N = X_train.shape[0]
        self.best_args['update_interval'] = update_interval
        self.best_args['y_dec'] = y_dec_train 
        self.best_args['roi_labels'] = y_train
        self.best_args['classes'] = classes
        self.best_args['batch_size'] = batch_size
        
        # selecting batch size
        # [42*t for t in range(42)]  will produce 16 train epochs
        # [0, 42, 84, 126, 168, 210, 252, 294, 336, 378, 420, 462, 504, 546, 588, 630]
        test_iter = mx.io.NDArrayIter({'data': X_train}, 
                                      batch_size=N, shuffle=False,
                                      last_batch_handle='pad')
        args = {k: mx.nd.array(v.asnumpy(), ctx=self.xpu) for k, v in self.args.items()}
        ## embedded point zi 
        z = model.extract_feature(self.feature, args, None, test_iter, N, self.xpu).values()[0]
        
        # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
        self.perplexity = 15
        self.learning_rate = 125
        # reconstruct wordy labels list(Y)==named_y
        named_y = [classes[kc] for kc in y_dec_train]
        self.best_args['named_y'] = named_y
        
        # To initialize the cluster centers, we pass the data through
        # the initialized DNN to get embedded data points and then
        # perform standard k-means clustering in the feature space Z
        # to obtain k initial centroids {mu j}
        kmeans = KMeans(self.best_args['num_centers'], n_init=20)
        kmeans.fit(z)
        args['dec_mu'][:] = kmeans.cluster_centers_
        
        ### KL DIVERGENCE MINIMIZATION. eq(2)
        # our model is trained by matching the soft assignment to the target distribution. 
        # To this end, we define our objective as a KL divergence loss between 
        # the soft assignments qi (pred) and the auxiliary distribution pi (label)
        solver = Solver('sgd', momentum=0.9, wd=0.0, learning_rate=0.01) # , lr_scheduler=mx.misc.FactorScheduler(20*update_interval,0.4)) #0.01
        def ce(label, pred):
            return np.sum(label*np.log(label/(pred+0.000001)))/label.shape[0]
        solver.set_metric(mx.metric.CustomMetric(ce))

        label_buff = np.zeros((X_train.shape[0], self.best_args['num_centers']))
        train_iter = mx.io.NDArrayIter({'data': X_train}, {'label': label_buff}, batch_size=self.best_args['batch_size'],
                                       shuffle=False, last_batch_handle='roll_over')
        self.best_args['y_pred'] = np.zeros((X_train.shape[0]))
        self.best_args['acci'] = []
        self.best_args['bestacci'] = []
        self.ploti = 0
        figprogress = plt.figure(figsize=(20, 15))  
        figROCs = plt.figure(figsize=(20, 15))                  
        print 'Batch_size = %f'% self.best_args['batch_size']
        print 'update_interval = %f'%  update_interval
        self.best_args['plot_interval'] = int(20*update_interval)
        print 'plot_interval = %f'%  self.best_args['plot_interval']
        self.maxAcc = 0.0
        
        def refresh(i): # i=3, a full epoch occurs every i=798/48
            if i%self.best_args['update_interval'] == 0:
                z = list(model.extract_feature(self.feature, args, None, test_iter, N, self.xpu).values())[0]
                                
                p = np.zeros((z.shape[0], self.best_args['num_centers']))
                self.dec_op.forward([z, args['dec_mu'].asnumpy()], [p])
                # the soft assignments qi (pred)
                y_pred = p.argmax(axis=1)
                #print np.std(np.bincount(y_dec_train)), np.bincount(y_dec_train)
                print np.std(np.bincount(y_pred)), np.bincount(y_pred)
                
                #####################
                # Z-space CV RF classfier METRICS
                #####################
                # compare soft assignments with known labels (only B or M)
                print '\n... Updating i = %f' % i      
                allL = np.asarray(y_train)
                Xdata = z[allL!='K',:]
                ydatalabels = np.asarray(allL[allL!='K']=='M').astype(int) # malignant is positive class
                
                cv = StratifiedKFold(n_splits=5)
                RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=0)
                # Evaluate a score by cross-validation
                tprs = []; aucs = []
                mean_fpr = np.linspace(0, 1, 100)
                cvi = 0
                for train, test in cv.split(Xdata, ydatalabels):
                    probas = RFmodel.fit(Xdata[train], ydatalabels[train]).predict_proba(Xdata[test])
                    # Compute ROC curve and area the curve
                    fpr, tpr, thresholds = roc_curve(ydatalabels[test], probas[:, 1])
                    # to create an ROC with 100 pts
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                    cvi += 1
               
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                                    
                # integer=5, to specify the number of folds in a (Stratified)KFold,
                #scores_BorM = cross_val_score(RFmodel, data, datalabels, cv=5)
                # compute Z-space Accuracy
                #Acc = scores_BorM.mean()
                Acc = mean_auc                       
                print "cvRF BorM Accuracy = %f " % Acc
                #print scores_BorM.tolist()
                    
                if(i==0):
                    tsne = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
                                init='pca', random_state=0, verbose=2, method='exact')
                    Z_tsne = tsne.fit_transform(z)        
                    self.best_args['initAcc'] = Acc
                    # plot initial z        
                    figinint = plt.figure()
                    axinint = figinint.add_subplot(1,1,1)
                    plot_embedding_unsuper_NMEdist_intenh(Z_tsne, named_y, axinint, title='kmeans init tsne: Acc={}'.format(Acc), legend=True)
                    figinint.savefig('{}//tsne_init_z{}_mu{}_{}.pdf'.format(save_to,self.best_args['znum'],self.best_args['num_centers'],labeltype), bbox_inches='tight')     
                    plt.close()                  
                    
                # save best args
                self.best_args['acci'].append( Acc )
                if(Acc >= self.maxAcc):
                    print 'Improving maxAcc = {}'.format(Acc)
                    for key, v in args.items():
                        self.best_args[key] = args[key]
                        
                    self.maxAcc = Acc
                    self.best_args['pbestacci'] = p
                    self.best_args['zbestacci']  = z 
                    self.best_args['bestacci'].append( Acc )
                    self.best_args['dec_mu'][:] = args['dec_mu'].asnumpy()
                
                if(i>0 and i%self.best_args['plot_interval']==0 and self.ploti<=15): 
                    # Visualize the progression of the embedded representation in a subsample of data
                    # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
                    tsne = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
                         init='pca', random_state=0, verbose=2, method='exact')
                    Z_tsne = tsne.fit_transform(z)
                    axprogress = figprogress.add_subplot(4,4,1+self.ploti)
                    plot_embedding_unsuper_NMEdist_intenh(Z_tsne, named_y, axprogress, title="Epoch %d z_tsne Acc (%f)" % (i,Acc), legend=False)
                    
                    # plot AUC
                    allL = np.asarray(y_train)
                    Xdata = z[allL!='K',:]
                    ydatalabels = np.asarray(allL[allL!='K']=='M').astype(int) # malignant is positive class
                    
                    cv = StratifiedKFold(n_splits=5)
                    RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=1000, random_state=0, verbose=0)
                    # Evaluate a score by cross-validation
                    tprs = []; aucs = []
                    mean_fpr = np.linspace(0, 1, 100)
                    cvi = 0
                    for train, test in cv.split(Xdata, ydatalabels):
                        probas = RFmodel.fit(Xdata[train], ydatalabels[train]).predict_proba(Xdata[test])
                        # Compute ROC curve and area the curve
                        fpr, tpr, thresholds = roc_curve(ydatalabels[test], probas[:, 1])
                        # to create an ROC with 100 pts
                        tprs.append(interp(mean_fpr, fpr, tpr))
                        tprs[-1][0] = 0.0
                        roc_auc = auc(fpr, tpr)
                        aucs.append(roc_auc)
                        # plot
                        axaroc = figROCs.add_subplot(4,4,1+self.ploti)
                        axaroc.plot(fpr, tpr, lw=1, alpha=0.5) # with label add: label='cv %d, AUC %0.2f' % (cvi, roc_auc)
                        cvi += 1
                        
                    axaroc.plot([0, 1], [0, 1], linestyle='--', lw=1, color='b',
                             label='Luck', alpha=.8)
                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    mean_auc = auc(mean_fpr, mean_tpr)
                    std_auc = np.std(aucs)
                    axaroc.plot(mean_fpr, mean_tpr, color='b',
                             label=r'MeanROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                             lw=2, alpha=.8)     
                    std_tpr = np.std(tprs, axis=0)
                    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                    axaroc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                                     label=r'$\pm$ 1 std. dev.') 
                    axaroc.set_xlim([-0.05, 1.05])
                    axaroc.set_ylim([-0.05, 1.05])
                    axaroc.set_xlabel('False Positive Rate')
                    axaroc.set_ylabel('True Positive Rate')
                    axaroc.set_title('ROC Epoch {}'.format(i))
                    axaroc.legend(loc="lower right")
                    
                    self.ploti = self.ploti+1
                      
                ## COMPUTING target distributions P
                ## we compute pi by first raising qi to the second power and then normalizing by frequency per cluster:
                weight = 1.0/p.sum(axis=0) # p.sum provides fj
                weight *= self.best_args['num_centers']/weight.sum()
                p = (p**2)*weight
                train_iter.data_list[1][:] = (p.T/p.sum(axis=1)).T
                print np.sum(y_pred != self.best_args['y_pred']), 0.001*y_pred.shape[0]
                
                # For the purpose of discovering cluster assignments, we stop our procedure when less than tol% of points change cluster assignment between two consecutive iterations.
                # tol% = 0.001
                if i == self.best_args['update_interval']*300: # performs 1epoch = 615/3 = 205*1000epochs                     
                    self.best_args['y_pred'] = y_pred   
                    self.best_args['acci'].append( Acc )
                    return True 
                    
                self.best_args['y_pred'] = y_pred

        # start solver
        solver.set_iter_start_callback(refresh)
        solver.set_monitor(Monitor(20))
        solver.solve(self.xpu, self.loss, args, self.args_grad, None,
                     train_iter, 0, 1000000000, {}, False)
        self.end_args = args
        self.best_args['end_args'] = args
        
        # finish                
        figprogress = plt.gcf()
        figprogress.savefig('{}\\tsne_progress_z{}_mu{}_{}.pdf'.format(save_to,self.best_args['znum'],self.best_args['num_centers'],labeltype), bbox_inches='tight')    
        plt.close()    
        
         # plot final z        
        figfinal = plt.figure()
        axfinal = figfinal.add_subplot(1,1,1)
        tsne = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
             init='pca', random_state=0, verbose=2, method='exact')
        Z_tsne = tsne.fit_transform(self.best_args['zbestacci'])      
        plot_embedding_unsuper_NMEdist_intenh(Z_tsne, self.best_args['named_y'], axfinal, title='final tsne: Acc={}'.format(self.best_args['bestacci'][-1]), legend=True)
        figfinal.savefig('{}\\tsne_final_z{}_mu{}_{}.pdf'.format(save_to,self.best_args['znum'],self.best_args['num_centers'],labeltype), bbox_inches='tight')    
        plt.close()          

        outdict = {'initAcc':self.best_args['initAcc'],
                   'acci': self.best_args['acci'],
                   'bestacci': self.best_args['bestacci'],
                    'pbestacci':self.best_args['pbestacci'],
                    'zbestacci':self.best_args['zbestacci'],
                    'dec_mubestacci':self.best_args['dec_mu'],
                    'y_pred': self.best_args['y_pred'],
                    'named_y': self.best_args['named_y'],
                    'classes':self.best_args['classes'],
                    'num_centers': self.best_args['num_centers'],
                    'znum':self.best_args['znum'],
                    'update_interval':self.best_args['update_interval'],
                    'batch_size':self.best_args['batch_size']}  
                             
        return outdict
        
            
if __name__ == '__main__':
    #####################################################
    from decModel_exp5 import *
    from utilities import *
    
    ## 1) read in the datasets both all NME (to do pretraining)
    NME_nxgraphs = r'Z:\Cristina\Section3\NME_DEC\imgFeatures\NME_nxgraphs'
    
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
       
    # shape input (798L, 427L)    
    combX_allNME = nxGdiscfeatures
    YnxG_allNME = np.asarray([nxGdatafeatures['roi_id'].values,
            nxGdatafeatures['classNME'].values,
            nxGdatafeatures['nme_dist'].values,
            nxGdatafeatures['nme_int'].values])
            
    print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
    print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0])   )
             
    ######################
    ## 2) DEC using labeled cases
    ######################                                            
    labeltype = 'classNME_onlynxG' 
    save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_exp5'   
    #log
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    LOG_FILE_stdout = os.path.join(save_to, 'allDEC_batch_exp5_{}.txt'.format(labeltype))
    #log to console    
    console_formatter = logging.Formatter('%(asctime)s %(message)s')
    #ch = logging.StreamHandler()
    ch = logging.FileHandler(LOG_FILE_stdout,'a')
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)
        
    # dfine num_centers according to clustering variable
    ## use y_dec to  minimizing KL divergence for clustering with known classes
    ysup = ["{}_{}_{}".format(a, b, c) if b!='nan' else "{}_{}".format(a, c) for a, b, c in zip(YnxG_allNME[1], YnxG_allNME[2], YnxG_allNME[3])]
    #ysup[range(combX_filledbyBC.shape[0])] = YnxG_filledbyBC[1]+'_'+YnxG_filledbyBC[3]+'_'+YnxG_filledbyBC[4] # +['_'+str(yl) for yl in YnxG_filledbyBC[3]]  
    #ysup[range(combX_filledbyBC.shape[0])] = YnxG_filledbyBC[1]+'_'+YnxG_filledbyBC[3]
    ysup = ['K'+rl[1::] if rl[0]=='U' else rl for rl in ysup]
    roi_labels = YnxG_allNME[1]  
    roi_labels = ['K' if rl=='U' else rl for rl in roi_labels]
    try:
        y_dec = np.asarray([int(label) for label in ysup])
    except:
        classes = [str(c) for c in np.unique(ysup)]
        numclasses = [i for i in range(len(classes))]
        y_dec = []
        for k in range(len(ysup)):
            for j in range(len(classes)):
                if(str(ysup[k])==classes[j]): 
                    y_dec.append(numclasses[j])
        y_dec = np.asarray(y_dec)
    
    ########################################################
    # DEC
    ########################################################
    input_size = combX_allNME.shape[1]
    latent_size = [input_size/rxf for rxf in [10,5]]
    varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(3,10,8)]
    
    for num_centers in varying_mu:
        initAccuracy = []
        cvRFZspaceAccuracy = []           
        normalizedMI = []
        cvRFOriginalXAccuracy = []

        # to load a prevously DEC model
        for znum in latent_size:
            # batch normalization
            sep = int(combX_allNME.shape[0]*0.10)
            X_val = combX_allNME[:sep]
            y_val = roi_labels[:sep]
            X_train = combX_allNME[sep:]
            y_dec_train = y_dec[sep:]
            y_train = roi_labels[sep:]
            batch_size = 125 # 160 
            update_interval = 6
        
            #num_centers = len(classes)
            # Read autoencoder: note is not dependent on number of clusters just on z latent size
            print "Load autoencoder of znum = ",znum
            print "Training DEC num_centers = ",num_centers
            logger.info('Load autoencoder of znum = {}, mu = {} \n Training DEC'.format(znum,num_centers))
            dec_model = DECModel(mx.cpu(), X_train, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 
            logger.info('Tunning DEC batch_size ={}, alpha anheling={}'.format(batch_size,update_interval)) # orig paper 256*40 (10240) point for upgrade about 1/6 (N) of data
            outdict = dec_model.cluster(X_train, y_dec_train, y_train, classes, batch_size, save_to, labeltype, update_interval) # 10 epochs# ~ per 1/3 of data 798/48=16 update twice per epoch ~ N/(batch size)=iterations to reach a full epochg
            #
            logger.info('Finised trainining DEC...') 
            print 'dec_model bestacci = {}'.format( outdict['bestacci'][-1] )
            logger.info('dec_model bestacci = {}'.format( outdict['bestacci'][-1] ))
            print 'dec_model initAcc = {}'.format( outdict['initAcc'] )
            logger.info('dec_model initAcc = {}'.format( outdict['initAcc'] ))
            
            print '5nn BorM_diag_Acc = {}'.format( outdict['BorM_diag_Acc'] )
            logger.info('5nn BorM_diag_Acc = {}'.format( outdict['BorM_diag_Acc'] ))
            print 'TPR = {}'.format( outdict['TPR'] )
            logger.info('TPR = {}'.format( outdict['TPR'] ))
            print 'TNR = {}'.format( outdict['TNR'] )
            logger.info('TNR = {}'.format( outdict['TNR'] ))
            print 'FPR = {}'.format( outdict['FPR'] )
            logger.info('FPR = {}'.format( outdict['FPR'] ))
            print 'FNR = {}'.format( outdict['FNR'] )
            logger.info('FNR = {}'.format( outdict['FNR'] ))
            print 'missedR = {}'.format( outdict['missedR'] )
            logger.info('missedR = {}'.format( outdict['missedR'] ))
    
            # save to plot as a function of znum        
            initAccuracy.append( outdict['initAcc'] )
            cvRFZspaceAccuracy.append( outdict['bestacci'][-1])
            BorM_diag_bestAcc.append( outdict['BorM_diag_Acc'] )
        
            # save output results
            dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
            'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
            'encoder_3_bias', 'encoder_2_bias']
            dec_args = {key: v.asnumpy() for key, v in dec_model.best_args.items() if key in dec_args_keys}
            dec_args['dec_mubestacci'] = dec_model.best_args['dec_mu']
            args_save = {key: v for key, v in dec_model.best_args.items() if key not in dec_args_keys}
            dec_model = dec_args.copy()
            dec_model.update(args_save) 
            fname = save_to+os.sep+'args_save_z{}_mu{}.arg'.format(znum,num_centers)
            with open(fname, 'w') as fout:
                pickle.dump(args_save, fout)
            # save model saving params into a numpy array
            dec_model_save= gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'wb')
            pickle.dump(dec_model, dec_model_save, protocol=pickle.HIGHEST_PROTOCOL)
            dec_model_save.close()
            
            ## plot iterations
            df = pd.DataFrame({'acc': pd.Series(outdict['acci']), 'iterations':range(len(outdict['acci']))})   
            fig2 = plt.figure(figsize=(20,6))
            ax2 = plt.axes()
            sns.set_context("notebook")
            sns.set_style("darkgrid", {"axes.facecolor": ".9"})    
            sns.pointplot(x="iterations", y="acc", data=df, ax=ax2, size=0.05) 
            fig2.autofmt_xdate(bottom=0.2, rotation=30, ha='right')   
            fig2.savefig(save_to+os.sep+'DEC_z{}_mu{}_{}-unsuprv acc vs iteration.pdf'.format(znum,num_centers,labeltype), bbox_inches='tight')    
            plt.close(fig2)
                    
            #####################
            # algorithm to find the epoch at best acc 
            #####################
            max_acc = 0
            epochs = []; best_acc = []
            for k,acc in enumerate(outdict['acci']):
                if(acc >= max_acc):
                    epochs.append(k)
                    best_acc.append(acc)
                    max_acc = acc
            # to plot best acci
            outdict['max_acc'] = max_acc
            logger.info('dec_model max_acc={}'.format(max_acc))
    
            #####################
            # Calculate normalized MI: find the relative frequency of points in Wk and Cj
            #####################
            N = X_train.shape[0]
            num_classes = len(np.unique(roi_labels)) # present but not needed during AE training
            roi_classes = np.unique(roi_labels)
            roi_labels = np.asarray(roi_labels)
            
            # extact embedding space
            all_iter = mx.io.NDArrayIter({'data': X_train}, batch_size=X_train.shape[0], shuffle=False,
                                                      last_batch_handle='pad')   
            ## embedded point zi 
            aDEC = DECModel(mx.cpu(), X_train, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 
            mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
            zbestacci = model.extract_feature(aDEC.feature, mxdec_args, None, all_iter, X_train.shape[0], aDEC.xpu).values()[0]      
            # orig paper 256*40 (10240) point for upgrade about 1/6 (N) of data
            #zbestacci = dec_model['zbestacci'] 
            pbestacci = np.zeros((zbestacci.shape[0], dec_model['num_centers']))
            aDEC.dec_op.forward([zbestacci, dec_args['dec_mubestacci'].asnumpy()], [pbestacci])
    
            # find max soft assignments dec_args
            W = pbestacci.argmax(axis=1)
            clusters = np.unique(W)
            num_clusters = len(np.unique(W))
            
            MLE_kj = np.zeros((num_clusters,num_classes))
            absWk = np.zeros((num_clusters))
            absCj = np.zeros((num_classes))
            for k in range(num_clusters):
                # find poinst in cluster k
                absWk[k] = np.sum(W==k)
                for j in range(num_classes):
                    # find points of class j
                    absCj[j] = np.sum(roi_labels==roi_classes[j])
                    # find intersection 
                    ptsk = W==k
                    MLE_kj[k,j] = np.sum(ptsk[roi_labels==roi_classes[j]])
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
            Hw = np.sum(absWk/N*np.log(absWk/N))
            NMI = Iwk/(np.abs(Hc+Hw)/2.0)
            print "... DEC normalizedMI = ", NMI
            # to plot best acci
            normalizedMI.append( NMI ) 
            outdict['NMI'] = NMI
            logger.info('dec_model NMI={}'.format(NMI))            
          
            ######## train an RF on ORIGINAL space Finally perform a cross-validation using RF
            datalabels = YnxG_allNME[1]
            data = X_train[datalabels!='K',:]
            datalabels = datalabels[datalabels!='K']
            RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=1)
            # Evaluate a score by cross-validation
            # integer=5, to specify the number of folds in a (Stratified)KFold,
            scores = cross_val_score(RFmodel, data, datalabels, cv=5)
            print "(cvRF Accuracy = %f " % scores.mean()      
            print scores.tolist()
            
            #appends and logs
            cvRFOriginalXAccuracy.append(scores.mean())
            outdict['cvRFOriginalXAccuracy'] = scores.mean()                 
            logger.info('mean 5cv cv OriginalX RF_Accuracy ={}'.format( scores.mean() ))
            logger.info('all ={}'.format( scores.tolist() ))
    
            # save model saving params into a numpy array
            outdict_save= gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'wb')
            pickle.dump(outdict, outdict_save, protocol=pickle.HIGHEST_PROTOCOL)
            outdict_save.close()    
    
        ########################################
        ##### finish plotting by each znum
        # plot latent space Accuracies vs. original
        colors = plt.cm.jet(np.linspace(0, 1, 16))
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(latent_size, initAccuracy, color=colors[6], ls=':', label='initAccuracy')
        ax1.plot(latent_size, cvRFZspaceAccuracy, color=colors[0], label='max_cvRF_Zspace')
        ax1.plot(latent_size, BorM_diag_bestAcc, color=colors[2], ls='-.', label='BorM_diag_bestAcc')  ## Average malignant and  benigng classs
        ax1.plot(latent_size, cvRFOriginalXAccuracy, color=colors[4], ls='-.', label='OriginalX_Acc_Malignant&Benign')
        ax1.plot(latent_size, normalizedMI, color=colors[8], label='normalizedMI')
        h1, l1 = ax1.get_legend_handles_labels()
        ax1.legend(h1, l1, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':10})
    
        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(latent_size, initAccuracy, color=colors[7], label='cvRFZspaceNME_DISTAcc')
        ax2.plot(latent_size, cvRFZspaceAccuracy, color=colors[12], label='cvRFZspaceNME_INTENHAcc')
        ax2.plot(latent_size, NME_DISTAcc, color=colors[7], ls='-.', label='NME_DISTAcc')  
        ax2.plot(latent_size, NME_INTENHAcc, color=colors[12], ls='-.', label='NME_INTENHAcc')
        ax2.plot(latent_size, cvRFNME_DISTAcc, color=colors[7], ls=':', label='OriginalX_cvRFNME_DISTAcc')  
        ax2.plot(latent_size, cvRFNME_INTENHAcc, color=colors[12], ls=':', label='OriginalX_cvRFNME_INTENHAcc')
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h2, l2, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':10})
        fig.savefig(save_to+os.sep+'allDEC_Accu_NMI_mu{}_{}-unsuprv acc vs iteration.pdf'.format(num_centers,labeltype), bbox_inches='tight')    
    
    # save to R
#    pdzfinal = pd.DataFrame( np.append( y[...,None], zfinal, 1) )
#    pdzfinal.to_csv('datasets//zfinal.csv', sep=',', encoding='utf-8', header=False, index=False)
#    # to save to csv
#    pdcombX = pd.DataFrame( np.append( y[...,None], combX, 1) )
#    pdcombX.to_csv('datasets//combX.csv', sep=',', encoding='utf-8', header=False, index=False)
#        


