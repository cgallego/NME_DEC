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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
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
        ae_model.load( os.path.join(save_to,'SAE_zsize{}_wimgfeatures_descStats.arg'.format(str(znum))) ) #_Nbatch_wimgfeatures
        logging.log(logging.INFO, "Reading Autoencoder from file..: %s"%(os.path.join(save_to,'SAE_zsize{}_wimgfeatures_descStats..arg'.format(znum))) )
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

    def cluster(self, X_train, y_dec_train, y_train, classes, batch_size, save_to, labeltype, update_interval, logger):
        N = X_train.shape[0]
        self.best_args['update_interval'] = update_interval
        self.best_args['y_dec'] = y_dec_train 
        self.best_args['roi_labels'] = y_train
        self.best_args['classes'] = classes
        self.best_args['batch_size'] = batch_size
        self.logger = logger
        
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
        solver = Solver('sgd',learning_rate=0.1)   ### original: 0.01, try1: Solver('sgd', momentum=0.9, wd=0.0, learning_rate=0.000125, lr_scheduler=mx.misc.FactorScheduler(20*update_interval,0.5))  try 2: Solver('sgd', momentum=0.6, wd=0.05, learning_rate=0.00125, lr_scheduler=mx.misc.FactorScheduler(20*update_interval,0.5)) 
        def ce(label, pred):
            #            acc = mx.metric.Accuracy()
            #            acc.update(preds = predicts, labels = labels)
            #            print acc.get()
            #             -acc.get()[0][1]
            return np.sum(label*np.log(label/(pred+0.000001)))/label.shape[0]
            
        # Deeo learning metrics to minimize
        solver.set_metric(mx.metric.CustomMetric(ce))

        ### Define DEC training varialbes
        label_buff = np.zeros((X_train.shape[0], self.best_args['num_centers']))
        train_iter = mx.io.NDArrayIter({'data': X_train}, 
                                       {'label': label_buff}, 
                                       batch_size=self.best_args['batch_size'],
                                       shuffle=False, last_batch_handle='roll_over')
                                       
        
        figprogress = plt.figure(figsize=(20, 15))  
        print 'Batch_size = %f'% self.best_args['batch_size']
        print 'update_interval = %f'%  update_interval
        self.best_args['plot_interval'] = int(5*update_interval)
        print 'plot_interval = %f'%  self.best_args['plot_interval']
        self.maxAcc = 0.0
        self.best_args['y_pred'] = np.zeros((X_train.shape[0]))
        self.best_args['acci'] = []
        self.best_args['bestacci'] = []
        self.ploti = 0
        
        def refresh(i): # i=3, a full epoch occurs every i=798/48
            if i%self.best_args['update_interval'] == 0:
                z = list(model.extract_feature(self.feature, args, None, test_iter, N, self.xpu).values())[0]
                                
                p = np.zeros((z.shape[0], self.best_args['num_centers']))
                self.dec_op.forward([z, args['dec_mu'].asnumpy()], [p])
                # the soft assignments qi (pred)
                y_pred = p.argmax(axis=1)
                #print np.std(np.bincount(y_dec_train)), np.bincount(y_dec_train)
                print np.std(np.bincount(y_pred)), np.bincount(y_pred)
                
                if(i==0):
                    self.tsne = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
                                init='pca', random_state=0, verbose=2, method='exact')
                    self.Z_tsne = self.tsne.fit_transform(z)  
                
                # compare soft assignments with known labels (only B or M)
                print '\n... Updating i = %f' % i 
                
                #####################
                # Z-space MLP fully coneected layer for classification
                #####################
                # compare soft assignments with known labels (only B or M)
                print '\n... Updating  i = %f' % i   
                sep = int(z.shape[0]*0.10)
                print(z.shape)
                datalabels = np.asarray(y_train)
                dataZspace = np.concatenate((z, p), axis=1) #zbestacci #dec_model['zbestacci']   
                Z = dataZspace[datalabels!='K',:]
                y = datalabels[datalabels!='K']
                print(Z)
                                
                # Do a 5 fold cross-validation
                Z_test = Z[:sep]
                yZ_test = np.asanyarray(y[:sep]=='M').astype(int) 
                Z_train = Z[sep:]
                yZ_train = np.asanyarray(y[sep:]=='M').astype(int) 
                batch_size = 1
                print(Z_test.shape)
                print(Z_train.shape)
                
                # Run classifier with cross-validation and plot ROC curves
                cv = StratifiedKFold(n_splits=5)
                # Evaluate a score by cross-validation
                tprs = []; aucs = []
                mean_fpr = np.linspace(0, 1, 100)
                cvi = 0
                for train, test in cv.split(Z_train, yZ_train):
                    # Multilayer Perceptron
                    MLP_train_iter = mx.io.NDArrayIter(Z_train[train], yZ_train[train],
                                                        batch_size, shuffle=True)
                    MLP_val_iter = mx.io.NDArrayIter(Z_train[test], yZ_train[test], batch_size)    
                    
                    # We’ll define the MLP using MXNet’s symbolic interface
                    dataMLP = mx.sym.Variable('data')
                    
                    #The following code declares two fully connected layers with 128 and 64 neurons each. 
                    #Furthermore, these FC layers are sandwiched between ReLU activation layers each 
                    #one responsible for performing an element-wise ReLU transformation on the FC layer output.
                    # The first fully-connected layer and the corresponding activation function
                    fc1  = mx.sym.FullyConnected(data=dataMLP, num_hidden = 128)
                    act1 = mx.sym.Activation(data=fc1, act_type="tanh")
                    
                    # data has 2 classes
                    fc2  = mx.sym.FullyConnected(data=act1, num_hidden=2)
                    # Softmax with cross entropy loss
                    mlp  = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
                    
                    # create a trainable module on CPU
                    mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
                    mlp_model.fit(MLP_train_iter,  # train data
                                  eval_data = MLP_val_iter,  # validation data
                                  optimizer='sgd',  # use SGD to train
                                  optimizer_params={'learning_rate':0.01},  # use fixed learning rate
                                  eval_metric= 'acc', #MLPacc(yZ_val, Z_val),  # report accuracy during training
                                  num_epoch=100)  # train for at most 10 dataset passes
    
                    #After the above training completes, we can evaluate the trained model by running predictions on validation data. 
                    #The following source code computes the prediction probability scores for each validation data. 
                    # prob[i][j] is the probability that the i-th validation contains the j-th output class.
                    prob_val = mlp_model.predict(MLP_val_iter)
                    # Compute ROC curve and area the curve
                    fpr, tpr, thresholds = roc_curve(yZ_train[test], prob_val.asnumpy()[:,1])
                    # to create an ROC with 100 pts
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    print roc_auc
                    aucs.append(roc_auc)
                    cvi += 1
                    
                # compute across all cvs
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs)
                print r'cv meanROC (AUC = {0:.4f} $\pm$ {0:.4f})'.format(mean_auc, std_auc)

                Z_test_iter = mx.io.NDArrayIter(Z_test,  None, batch_size)
                prob_test = mlp_model.predict(Z_test_iter)
                # Compute ROC curve and area the curve
                fpr_val, tpr_val, thresholds_val = roc_curve(yZ_test, prob_test.asnumpy()[:,1])
                auc_val = auc(fpr_val, tpr_val)
                print r'cv test (AUC = {0:.4f})'.format(auc_val)

                # compute Z-space Accuracy
                self.Acc = mean_auc
                
                if(i==0):
                    self.best_args['initAcc'] = self.Acc
                    # plot initial z        
                    figinint = plt.figure()
                    axinint = figinint.add_subplot(1,1,1)
                    plot_embedding_unsuper_NMEdist_intenh(self.Z_tsne, named_y, axinint, title='kmeans init tsne: AUC={}\n'.format(self.Acc), legend=True)
                    figinint.savefig('{}//tsne_init_z{}_mu{}_{}.pdf'.format(save_to,self.best_args['znum'],self.best_args['num_centers'],labeltype), bbox_inches='tight')     
                    plt.close() 
                
                # save best args
                self.best_args['acci'].append( self.Acc )
                if(self.Acc >= self.maxAcc):
                    print 'Improving maxAUC = {0:.4f}'.format(self.Acc)
                    self.logger.info('Improving maxAUC = {0:.4f} \n'.format(self.Acc))
            
                    for key, v in args.items():
                        self.best_args[key] = args[key]
                        
                    self.maxAcc = self.Acc
                    self.best_args['pbestacci'] = p
                    self.best_args['zbestacci']  = z 
                    self.best_args['Z_tsnebestacci'] = self.Z_tsne
                    self.best_args['bestacci'].append( self.Acc )
                    self.best_args['dec_mu'][:] = args['dec_mu'].asnumpy()
                
                if(i>0 and i%self.best_args['plot_interval']==0 and self.ploti<=15): 
                    # Visualize the progression of the embedded representation in a subsample of data
                    # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
                    tsne = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
                         init='pca', random_state=0, verbose=2, method='exact')
                    Z_tsne = tsne.fit_transform(z)
                    axprogress = figprogress.add_subplot(4,4,1+self.ploti)
                    plot_embedding_unsuper_NMEdist_intenh(Z_tsne, named_y, axprogress, title="Epoch %d z_tsne AUC (%f)" % (i,self.Acc), legend=False)
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
                if i == self.best_args['update_interval']*100: # performs 1epoch = 615/3 = 205*1000epochs                     
                    self.best_args['y_pred'] = y_pred   
                    self.best_args['acci'].append( Acc )
                    return True 
                    
                self.best_args['y_pred'] = y_pred

        # start solver
        solver.set_iter_start_callback(refresh)
        solver.set_monitor(Monitor(10))
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
        plot_embedding_unsuper_NMEdist_intenh(Z_tsne, self.best_args['named_y'], axfinal, title='final tsne: Acc={}\n'.format(self.best_args['bestacci'][-1]), legend=True)
        figfinal.savefig('{}\\tsne_final_z{}_mu{}_{}.pdf'.format(save_to,self.best_args['znum'],self.best_args['num_centers'],labeltype), bbox_inches='tight')    
        plt.close()          

        outdict = {'initAcc':self.best_args['initAcc'],
                   'acci': self.best_args['acci'],
                   'bestacci': self.best_args['bestacci'],
                    'pbestacci':self.best_args['pbestacci'],
                    'zbestacci':self.best_args['zbestacci'],
                    'Z_tsnebestacci':self.best_args['Z_tsnebestacci'],
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
    from decModel_wimgF_descStats import *
    from utilities import *
    
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
    x_min, x_max = np.min(allNMEs_dynamic, 0), np.max(allNMEs_dynamic, 0)
    x_max[x_max==0]=1.0e-07
    normdynamic = (allNMEs_dynamic - x_min) / (x_max - x_min)
    print(np.min(normdynamic, 0))
    print(np.max(normdynamic, 0))
    
    print('Normalizing morphology {} leasions with features of size = {}'.format(allNMEs_morphology.shape[0], allNMEs_morphology.shape[1]))
    x_min, x_max = np.min(allNMEs_morphology, 0), np.max(allNMEs_morphology, 0)
    x_max[x_max==0]=1.0e-07
    normorpho = (allNMEs_morphology - x_min) / (x_max - x_min)
    print(np.min(normorpho, 0))
    print(np.max(normorpho, 0))
     
    print('Normalizing texture {} leasions with features of size = {}'.format(allNMEs_texture.shape[0], allNMEs_texture.shape[1]))
    x_min, x_max = np.min(allNMEs_texture, 0), np.max(allNMEs_texture, 0)
    x_max[x_max==0]=1.0e-07
    normtext = (allNMEs_texture - x_min) / (x_max - x_min)
    print(np.min(normtext, 0))
    print(np.max(normtext, 0))
    
    print('Normalizing stage1 {} leasions with features of size = {}'.format(allNMEs_stage1.shape[0], allNMEs_stage1.shape[1]))
    x_min, x_max = np.min(allNMEs_stage1, 0), np.max(allNMEs_stage1, 0)
    x_min[np.isnan(x_min)]=1.0e-07
    x_max[np.isnan(x_max)]=1.0
    normstage1 = (allNMEs_stage1 - x_min) / (x_max - x_min)
    normstage1[np.isnan(normstage1)]=1.0e-07
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
    ## 2) DEC using labeled cases
    ######################                                            
    labeltype = 'wimgF_descStats' 
    save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_descStats'
   
    #log
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    LOG_FILE_stdout = os.path.join(save_to, 'decModel_{}.txt'.format(labeltype))
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
    latent_size = [input_size/rxf for rxf in [5]]
    varying_mu = [6] #[int(np.round(var_mu)) for var_mu in np.linspace(3,10,8)]
    
    for znum in latent_size:
        initAccuracy = []
        cvRFZspaceAccuracy = []           
        normalizedMI = []
        cvRFOriginalXAccuracy = []
        
        # to load a prevously DEC model
        for num_centers in varying_mu: 
            # batch normalization
            X_train = combX_allNME
            y_dec_train = y_dec
            y_train = roi_labels
            batch_size = X_train.shape[0]
            update_interval = 5 # approx. 4 epochs per update
            #            if(num_centers==3 and znum==30):
            #                continue
            
            #num_centers = len(classes)
            # Read autoencoder: note is not dependent on number of clusters just on z latent size
            print "Load autoencoder of znum = ",znum
            print "Training DEC num_centers = ",num_centers
            logger.info('Load autoencoder of znum = {}, mu = {} \n Training DEC'.format(znum,num_centers))
            logger.info('DEC optimization using learnRate = Solver(sgd, learning_rate=0.01) \n')
            logger.info('DEC batch_size = {}, update_interval = {} \n Training DEC'.format(batch_size,update_interval))
            
            dec_model = DECModel(mx.cpu(), X_train, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 
            logger.info('Tunning DEC batch_size ={}, alpha anheling={}'.format(batch_size,update_interval)) # orig paper 256*40 (10240) point for upgrade about 1/6 (N) of data
            outdict = dec_model.cluster(X_train, y_dec_train, y_train, classes, batch_size, save_to, labeltype, update_interval, logger) # 10 epochs# ~ per 1/3 of data 798/48=16 update twice per epoch ~ N/(batch size)=iterations to reach a full epochg
            #
            logger.info('Finised trainining DEC...') 
            print 'dec_model initAcc = {}'.format( outdict['initAcc'] )
            logger.info('dec_model initAcc = {}'.format( outdict['initAcc'] ))

            print 'dec_model bestacci = {}'.format( outdict['bestacci'][-1] )
            logger.info('dec_model bestacci = {}'.format( outdict['bestacci'][-1] ))
            # save to plot as a function of znum        
            initAccuracy.append( outdict['initAcc'] )
            cvRFZspaceAccuracy.append( outdict['bestacci'][-1])
          
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
            y_train_roi_labels = np.asarray(y_train)
            
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
                    absCj[j] = np.sum(y_train_roi_labels==roi_classes[j])
                    # find intersection 
                    ptsk = W==k
                    MLE_kj[k,j] = np.sum(ptsk[y_train_roi_labels==roi_classes[j]])
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
            datalabels = y_train_roi_labels
            Xdata = X_train[datalabels!='K',:]
            ydatalabels = datalabels[datalabels!='K']
            RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=500, random_state=0, verbose=1)
            # Evaluate a score by cross-validation
            # integer=5, to specify the number of folds in a (Stratified)KFold,
            scores = cross_val_score(RFmodel, Xdata, ydatalabels, cv=5)
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
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(varying_mu, initAccuracy, color=colors[6], ls=':', label='initAccuracy')
        ax1.plot(varying_mu, cvRFZspaceAccuracy, color=colors[0], label='max_cvRF_Zspace')
        ax1.plot(varying_mu, cvRFOriginalXAccuracy, color=colors[4], ls='-.', label='OriginalX_Acc_Malignant&Benign')
        ax1.plot(varying_mu, normalizedMI, color=colors[8], label='normalizedMI')
        h1, l1 = ax1.get_legend_handles_labels()
        ax1.legend(h1, l1, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':10})
    
    # save to R
#    pdzfinal = pd.DataFrame( np.append( y[...,None], zfinal, 1) )
#    pdzfinal.to_csv('datasets//zfinal.csv', sep=',', encoding='utf-8', header=False, index=False)
#    # to save to csv
#    pdcombX = pd.DataFrame( np.append( y[...,None], combX, 1) )
#    pdcombX.to_csv('datasets//combX.csv', sep=',', encoding='utf-8', header=False, index=False)
#        


