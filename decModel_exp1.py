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
try:
   import cPickle as pickle
except:
   import pickle
import gzip
   

def cluster_acc(Y_pred, Y):
    # For all algorithms we set the
    # number of clusters to the number of ground-truth categories
    # and evaluate performance with unsupervised clustering ac-curacy (ACC):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
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
        batch_size = 16
        ae_model = AutoEncoderModel(self.xpu, [X.shape[1],500,500,2000,znum], pt_dropout=0.2)
        ae_model.load( os.path.join(save_to,'SAE_zsize{}.arg'.format(str(znum))) )
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
        self.znum = znum
        self.batch_size = batch_size

    def cluster(self, X, y, classes, save_to, labeltype, update_interval=None):
        N = X.shape[0]
        self.update_interval = update_interval

        # selecting batch size
        # [42*t for t in range(42)]  will produce 16 train epochs
        # [0, 42, 84, 126, 168, 210, 252, 294, 336, 378, 420, 462, 504, 546, 588, 630]
        batch_size = self.batch_size #615/3 42  #256
        test_iter = mx.io.NDArrayIter({'data': X}, 
                                      batch_size=batch_size, shuffle=False,
                                      last_batch_handle='pad')
        args = {k: mx.nd.array(v.asnumpy(), ctx=self.xpu) for k, v in self.args.items()}
        ## embedded point zi 
        z = model.extract_feature(self.feature, args, None, test_iter, N, self.xpu).values()[0]
        
        # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
        self.perplexity = 15
        self.learning_rate = 95
        tsne = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
             init='pca', random_state=0, verbose=2, method='exact')
        Z_tsne = tsne.fit_transform(z)
        
        # plot initial z        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # reconstruct wordy labels list(Y)==named_y
        named_y = [classes[kc] for kc in y]
        plot_embedding(Z_tsne, named_y, ax, title='{} tsne with perplexity {}'.format(labeltype,self.perplexity), legend=True, plotcolor=True)
        fig.savefig(save_to+os.sep+'tsne_init_k'+str(labeltype)+'_z'+str(self.znum)+'.pdf', bbox_inches='tight')    
        plt.close()  
        
        # To initialize the cluster centers, we pass the data through
        # the initialized DNN to get embedded data points and then
        # perform standard k-means clustering in the feature space Z
        # to obtain k initial centroids {mu j}
        kmeans = KMeans(self.num_centers, n_init=20)
        kmeans.fit(z)
        args['dec_mu'][:] = kmeans.cluster_centers_
        
        ### KL DIVERGENCE MINIMIZATION. eq(2)
        # our model is trained by matching the soft assignment to the target distribution. 
        # To this end, we define our objective as a KL divergence loss between 
        # the soft assignments qi (pred) and the auxiliary distribution pi (label)
        solver = Solver('sgd', momentum=0.25, wd=0.0, learning_rate=0.0025) #0.01
        def ce(label, pred):
            return np.sum(label*np.log(label/(pred+0.000001)))/label.shape[0]
        solver.set_metric(mx.metric.CustomMetric(ce))

        label_buff = np.zeros((X.shape[0], self.num_centers))
        train_iter = mx.io.NDArrayIter({'data': X}, {'label': label_buff}, batch_size=batch_size,
                                       shuffle=False, last_batch_handle='roll_over')
        self.y_pred = np.zeros((X.shape[0]))
        self.acci = []
        self.ploti = 0
        fig = plt.figure(figsize=(20, 15))
        print 'Batch_size = %f'% self.batch_size
        print 'update_interval = %f'%  self.update_interval
        self.plot_interval = int(30*self.update_interval)
        print 'plot_interval = %f'%  self.plot_interval
        
        def refresh(i):
            if i%self.update_interval == 0:
                z = model.extract_feature(self.feature, args, None, test_iter, N, self.xpu).values()[0]
                                
                p = np.zeros((z.shape[0], self.num_centers))
                self.dec_op.forward([z, args['dec_mu'].asnumpy()], [p])
                # the soft assignments qi (pred)
                y_pred = p.argmax(axis=1)
                print np.std(np.bincount(y_pred)), np.bincount(y_pred)

                if y is not None:
                    # compare soft assignments with known labels (unused)
                    print '... Updating i = %f' % i 
                    print np.std(np.bincount(y.astype(np.int))), np.bincount(y.astype(np.int))
                    print y_pred[0:5], y.astype(np.int)[0:5]    
                    print 'Clustering Acc = %f'% cluster_acc(y_pred, y)[0]
                    self.acci.append( cluster_acc(y_pred, y)[0] )
                
                if(i%self.plot_interval==0 and self.ploti<=15): 
                    # Visualize the progression of the embedded representation in a subsample of data
                    # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
                    tsne = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
                         init='pca', random_state=0, verbose=2, method='exact')
                    Z_tsne = tsne.fit_transform(z)
                    
                    ax = fig.add_subplot(4,4,1+self.ploti)
                    plot_embedding(Z_tsne, named_y, ax, title="Epoch %d z_tsne iter (%d)" % (self.ploti,i), legend=False, plotcolor=True)
                    self.ploti = self.ploti+1
                    
                ## COMPUTING target distributions P
                ## we compute pi by first raising qi to the second power and then normalizing by frequency per cluster:
                weight = 1.0/p.sum(axis=0) # p.sum provides fj
                weight *= self.num_centers/weight.sum()
                p = (p**2)*weight
                train_iter.data_list[1][:] = (p.T/p.sum(axis=1)).T
                print np.sum(y_pred != self.y_pred), 0.001*y_pred.shape[0]
                
                # For the purpose of discovering cluster assignments, we stop our procedure when less than tol% of points change cluster assignment between two consecutive iterations.
                # tol% = 0.001
                if i == self.plot_interval*20: # performs 1epoch = 615/3 = 205*1000epochs                     
                    self.y_pred = y_pred
                    self.p = p
                    self.z = z                    
                    return True 
                    
                self.y_pred = y_pred
                self.p = p
                self.z = z

        # start solver
        solver.set_iter_start_callback(refresh)
        solver.set_monitor(Monitor(10))
        solver.solve(self.xpu, self.loss, args, self.args_grad, None,
                     train_iter, 0, 1000000000, {}, False)
        self.end_args = args
        
        # finish                
        fig = plt.gcf()
        fig.savefig(save_to+os.sep+'tsne_progress_k'+str(labeltype)+'_z'+str(self.znum)+'.pdf', bbox_inches='tight')    
        plt.close()    
        
         # plot final z        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        tsne = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
             init='pca', random_state=0, verbose=2, method='exact')
        Z_tsne = tsne.fit_transform(self.z)       
        plot_embedding(Z_tsne, named_y, ax, title="tsne with perplexity %d" % self.perplexity, legend=True, plotcolor=True)
        fig.savefig(save_to+os.sep+'tsne_final_k'+str(labeltype)+'_z'+str(self.znum)+'.pdf', bbox_inches='tight')    
        plt.close()  
        
        outdict = {'acc': self.acci,
                   'p': self.p,
                   'z': self.z,
                   'y_pred': self.y_pred,
                   'named_y': named_y}
            
        return outdict
        
            
if __name__ == '__main__':
    #####################################################
    from decModel_exp1 import *
    from utilities import *
    
    ## 1) read in the datasets both all NME (to do pretraining)
    NME_nxgraphs = r'Z:\Cristina\Section3\NME_DEC\imgFeatures\NME_nxgraphs'
    # start by loading nxGdatafeatures
    with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures.pklz'), 'rb') as fin:
        nxGdatafeatures = pickle.load(fin)
                     
    # to load SERw matrices for all lesions
    with gzip.open(os.path.join(NME_nxgraphs,'SER_edgesw_allNMEs_25binsize.pklz'), 'rb') as fin:
        alldiscrSERcounts = pickle.load(fin)
    
    # to load discrall_dict dict for all lesions
    with gzip.open(os.path.join(NME_nxgraphs,'discrall_dict_allNMEs_10binsize.pklz'), 'rb') as fin:
        discrall_dict_allNMEs = pickle.load(fin)           
   
    #########
    # exclude rich club bcs differnet dimenstions
    delRC = discrall_dict_allNMEs.pop('discrallDEL_rich_club')
    mstRC = discrall_dict_allNMEs.pop('discrallMST_rich_club')
    delsC = discrall_dict_allNMEs.pop('discrallMST_scluster')
    mstsC = discrall_dict_allNMEs.pop('discrallDEL_scluster')
    ########## for nxGdiscfeatures.shape = (202, 420)
    ds=discrall_dict_allNMEs.pop('DEL_dassort')
    ms=discrall_dict_allNMEs.pop('MST_dassort')
    # normalize 0-1    
    x_min, x_max = np.min(ds, 0), np.max(ds, 0)
    ds = (ds - x_min) / (x_max - x_min)
    x_min, x_max = np.min(ms, 0), np.max(ms, 0)
    ms = (ms - x_min) / (x_max - x_min)
    
    ## concatenate dictionary items into a nd array 
    ## normalize per x
    normgdiscf = []
    for fname,fnxg in discrall_dict_allNMEs.iteritems():
        print 'Normalizing.. {} \n min={}, \n max={} \n'.format(fname, np.min(fnxg, 0), np.max(fnxg, 0))
        x_min, x_max = np.min(fnxg, 0), np.max(fnxg, 0)
        x_max[x_max==0]=1.0e-07
        fnxg = (fnxg - x_min) / (x_max - x_min)
        normgdiscf.append( fnxg )
        print(np.min(fnxg, 0))
        print(np.max(fnxg, 0))
        
    nxGdiscfeatures = np.concatenate([gdiscf for gdiscf in normgdiscf], axis=1)
    # append other univariate features  nxGdiscfeatures.shape  (798L, 422L)               
    nxGdiscfeatures = np.concatenate((nxGdiscfeatures,                    
                                        ds.reshape(len(ds),1),
                                        ms.reshape(len(ms),1)), axis=1)
    # shape input 
    combX_allNME = np.concatenate((alldiscrSERcounts, nxGdiscfeatures), axis=1)       
    YnxG_allNME = [nxGdatafeatures['roi_label'].values,
            nxGdatafeatures['roiBIRADS'].values,
            nxGdatafeatures['NME_dist'].values,
            nxGdatafeatures['NME_int_enh'].values]
    
    print('Loading {} all NME of size = {}'.format(combX_allNME.shape[0], combX_allNME.shape[1]) )
    print('Loading all NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_allNME[0].shape[0])   )
    
    ################
    ## 1-b) read in the datasets both all NME and filledbyBC (to do finetunning)
    # to load nxGdatafeatures df for all lesions
    with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_filledbyBC.pklz'), 'rb') as fin:
        nxGdatafeatures = pickle.load(fin)
    # to load SERw matrices for all lesions
    with gzip.open(os.path.join(NME_nxgraphs,'SER_edgesw_filledbyBC.pklz'), 'rb') as fin:
        alldiscrSERcounts = pickle.load(fin)
    # to load discrall_dict dict for all lesions
    with gzip.open(os.path.join(NME_nxgraphs,'discrall_dict_filledbyBC.pklz'), 'rb') as fin:
        discrall_dict_filledbyBC = pickle.load(fin)
    
    ########
    # exclude rich club bcs differnet dimenstions
    delRC = discrall_dict_filledbyBC.pop('discrallDEL_rich_club')
    mstRC = discrall_dict_filledbyBC.pop('discrallMST_rich_club')
    delsC = discrall_dict_filledbyBC.pop('discrallMST_scluster')
    mstsC = discrall_dict_filledbyBC.pop('discrallDEL_scluster')
    ########## for nxGdiscfeatures.shape = (202, 420)
    ds=discrall_dict_filledbyBC.pop('DEL_dassort')
    ms=discrall_dict_filledbyBC.pop('MST_dassort')
    # normalize 0-1    
    x_min, x_max = np.min(ds, 0), np.max(ds, 0)
    ds = (ds - x_min) / (x_max - x_min)
    x_min, x_max = np.min(ms, 0), np.max(ms, 0)
    ms = (ms - x_min) / (x_max - x_min)
    
    ## concatenate dictionary items into a nd array 
    ## normalize per x
    normgdiscf = []
    for fname,fnxg in discrall_dict_filledbyBC.iteritems():
        print 'Normalizing.. {} \n min={}, \n max={} \n'.format(fname, np.min(fnxg, 0), np.max(fnxg, 0))
        x_min, x_max = np.min(fnxg, 0), np.max(fnxg, 0)
        x_max[x_max==0]=1.0e-07
        fnxg = (fnxg - x_min) / (x_max - x_min)
        normgdiscf.append( fnxg )
        print(np.min(fnxg, 0))
        print(np.max(fnxg, 0))
        
    nxGdiscfeatures = np.concatenate([gdiscf for gdiscf in normgdiscf], axis=1)
    # append other univariate features  nxGdiscfeatures.shape  (798L, 422L)               
    nxGdiscfeatures = np.concatenate((nxGdiscfeatures,                    
                                        ds.reshape(len(ds),1),
                                        ms.reshape(len(ms),1)), axis=1)
    # shape input 
    combX_filledbyBC = np.concatenate((alldiscrSERcounts, nxGdiscfeatures), axis=1)       
    YnxG_filledbyBC = [nxGdatafeatures['roi_label'].values,
            nxGdatafeatures['roiBIRADS'].values,
            nxGdatafeatures['NME_dist'].values,
            nxGdatafeatures['NME_int_enh'].values]

    print('Loading {} NME filled by BC of size = {}'.format(combX_filledbyBC.shape[0], combX_filledbyBC.shape[1]) )
    print('Loading NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_filledbyBC[0].shape[0])   )
                                              
    ######################
    ## 2) DEC using labeled cases
    ######################
    labeltype = 'roilabel_NME_dist' # 
    save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels'
    logging.basicConfig(filename=os.path.join(save_to,'allDEC_exp1_y{}.log'.format(labeltype)), format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)                    
    
    input_size = combX_filledbyBC.shape[1]
    latent_size = [input_size/rxf for rxf in [25,15,10,5]]
    batch_size = 32 # 160 32*5 = update_interval*5
    
    # dfine num_centers according to clustering variable
    ## use y_dec to  minimizing KL divergence for clustering with known classes
    Y = YnxG_filledbyBC[0]+'_'+YnxG_filledbyBC[2] # +['_'+str(yl) for yl in YnxG_filledbyBC[3]]  # YnxG_filledbyBC[0]+'_'+
    num_centers = len(np.unique(Y))
    try:
        y_dec = np.asarray([int(label) for label in Y])
    except:
        classes = [str(c) for c in np.unique(Y)]
        numclasses = [i for i in range(len(classes))]
        ynum_dec = Y
        y_dec = []
        for k in range(len(Y)):
            for j in range(len(classes)):
                if(str(Y[k])==classes[j]): 
                    y_dec.append(numclasses[j])
        y_dec = np.asarray(y_dec)
    
    ########################################################
    # DEC 
    ########################################################
    allDEC = []
    for znum in latent_size:
        # Read autoencoder: note is not dependent on number of clusters just on z latent size
        print "Building autoencoder of latent size znum = ",znum
        dec_model = DECModel(mx.cpu(), combX_filledbyBC, num_centers, 1.0, znum, save_to)
        outdict = dec_model.cluster(combX_filledbyBC, y_dec, classes, save_to, labeltype, update_interval=12) # ~ 202/16=6 ~ N/(batch size)=iterations to reach a full epochg
        # save output results
        dec_model.save( os.path.join(save_to,'dec_model_znum{}_exp1_y{}.arg'.format(znum,labeltype)) )
        allDEC.append(outdict)
        # to save                      
        logging.log(logging.INFO, "finished DEC training znum = "+str(znum))
    
    # save output
    allDEC_file = gzip.open(os.path.join(save_to,'allDEC_exp1_y{}.pklz'.format(labeltype)), 'wb')
    pickle.dump(allDEC, allDEC_file, protocol=pickle.HIGHEST_PROTOCOL)
    allDEC_file.close()
    
    #####################
    ## Visualize the reconstructed inputs and the encoded representations.
    ######################
    # train/test loss value o
    dfDEC_perf = pd.DataFrame()
    for DEC_perf in allDEC:
        dfDEC_perf = dfDEC_perf.append( pd.DataFrame({'acc': pd.Series(DEC_perf['acc']), 'iterations':range(len(DEC_perf['acc'])), 'Z-spaceDim':DEC_perf['z'].shape[1] }) )
        
    fig = plt.figure(figsize=(20,6))
    ax = plt.axes()
    sns.set_context("notebook")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})    
    sns.pointplot(x="iterations", y="acc", hue="Z-spaceDim", data=dfDEC_perf, size=0.25, plot_kws=dict(linewidth=0.5, edgecolor="w"), ax=ax)                               
    fig.savefig(save_to+os.sep+'allDEC_exp1_y{} Dataset_filledbyBC acc vs iteration.pdf'.format(labeltype), bbox_inches='tight')    

    ## to load saved variables
#    with gzip.open(os.path.join(save_to,'allDEC_exp1_y{}.pklz'.format(labeltype)), 'rb') as fu:
#        allDEC = pickle.load(fu)
        
     # save to R
    pdzfinal = pd.DataFrame( np.append( y[...,None], zfinal, 1) )
    pdzfinal.to_csv('datasets//zfinal.csv', sep=',', encoding='utf-8', header=False, index=False)
    # to save to csv
    pdcombX = pd.DataFrame( np.append( y[...,None], combX, 1) )
    pdcombX.to_csv('datasets//combX.csv', sep=',', encoding='utf-8', header=False, index=False)
        
        

        