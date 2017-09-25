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
        ae_model.load( os.path.join(save_to,'SAE_zsize{}_wimgfeatures.arg'.format(str(znum))) ) #_Nbatch_wimgfeatures
        logging.log(logging.INFO, "Reading Autoencoder from file..: %s"%(os.path.join(save_to,'SAE_zsize{}_wimgfeatures.arg'.format(znum))) )
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

    def cluster_varying_mu(self, X, y_dec, roi_labels, classes, batch_size, save_to, labeltype, update_interval=None):
        # y = y_dec X, y_dec, roi_labels, classes, batch_size, save_to, labeltype, update_in
        N = X.shape[0]
        self.best_args['update_interval'] = update_interval
        self.best_args['y_dec'] = y_dec 
        self.best_args['roi_labels'] = roi_labels
        self.best_args['classes'] = classes
        self.best_args['batch_size'] = batch_size
        
        # selecting batch size
        # [42*t for t in range(42)]  will produce 16 train epochs
        # [0, 42, 84, 126, 168, 210, 252, 294, 336, 378, 420, 462, 504, 546, 588, 630]
        test_iter = mx.io.NDArrayIter({'data': X}, 
                                      batch_size=N, shuffle=False,
                                      last_batch_handle='pad')
        args = {k: mx.nd.array(v.asnumpy(), ctx=self.xpu) for k, v in self.args.items()}
        ## embedded point zi 
        z = model.extract_feature(self.feature, args, None, test_iter, N, self.xpu).values()[0]
        
        # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
        self.perplexity = 25
        self.learning_rate = 200
        # reconstruct wordy labels list(Y)==named_y
        named_y = [classes[kc] for kc in y_dec]
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

        label_buff = np.zeros((X.shape[0], self.best_args['num_centers']))
        train_iter = mx.io.NDArrayIter({'data': X}, {'label': label_buff}, batch_size=self.best_args['batch_size'],
                                       shuffle=False, last_batch_handle='roll_over')
        self.best_args['y_pred'] = np.zeros((X.shape[0]))
        self.best_args['acci'] = []
        self.best_args['bestacci'] = []
        self.ploti = 0
        figprogress = plt.figure(figsize=(20, 15))                    
        print 'Batch_size = %f'% self.best_args['batch_size']
        print 'update_interval = %f'%  update_interval
        self.best_args['plot_interval'] = int(5*update_interval)
        print 'plot_interval = %f'%  self.best_args['plot_interval']
        self.maxAcc = 0.0
        
        def refresh(i): # i=3, a full epoch occurs every i=798/48
            if i%self.best_args['update_interval'] == 0:
                z = list(model.extract_feature(self.feature, args, None, test_iter, N, self.xpu).values())[0]
                                
                p = np.zeros((z.shape[0], self.best_args['num_centers']))
                self.dec_op.forward([z, args['dec_mu'].asnumpy()], [p])
                # the soft assignments qi (pred)
                y_pred = p.argmax(axis=1)
                print np.std(np.bincount(y_pred)), np.bincount(y_pred)

                # use a y that only considers the filledbyBC examples
                # compare soft assignments with known labels
                print '\n... Updating i = %f' % i      
                allL = np.asarray(roi_labels)
                data = z[allL!='K',:]
                datalabels = allL[allL!='K']
                RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=200, random_state=0, verbose=0)
                # Evaluate a score by cross-validation
                # integer=5, to specify the number of folds in a (Stratified)KFold,
                scores = cross_val_score(RFmodel, data, datalabels, cv=5)
               
                # do for overall class B and M
                labels = np.asarray(self.best_args['named_y'])
                Z_embedding_tree = sklearn.neighbors.BallTree(z, leaf_size=6)     
                # This finds the indices of 5 closest neighbors
                nme_dist = ['Diffuse', 'Focal', 'Linear', 'MultipleRegions', 'Regional','Segmental','N/A']
                nme_intenh = ['Clumped', 'ClusteredRing', 'Heterogeneous', 'Homogeneous','Stippled or punctate','N/A']
                wnme_dist = np.zeros((len(nme_dist),len(nme_dist)), dtype=np.int64)
                wnme_intenh = np.zeros((len(nme_intenh),len(nme_intenh)), dtype=np.int64)
                NME_descriptorate = []
                kznme = 0
                for k in range(z.shape[0]):
                    iclass = labels[k]
                    dist, ind = Z_embedding_tree.query([z[k]], k=6)
                    dist5nn, ind5nn = dist[k!=ind], ind[k!=ind]
                    class5nn = labels[ind5nn]
                    if(len(class5nn)>0 and (iclass.split('_')[1]!='N/A' or iclass.split('_')[1]!='N/A')):
                        # increment detections for final NME descriptor accuracy
                        kznme +=1
                        prednmed=[]
                        prednmed.append( sum([lab.split('_')[1]=='Diffuse' for lab in class5nn]) )
                        prednmed.append( sum([lab.split('_')[1]=='Focal' for lab in class5nn]) )
                        prednmed.append( sum([lab.split('_')[1]=='Linear' for lab in class5nn]) )
                        prednmed.append( sum([lab.split('_')[1]=='MultipleRegions' for lab in class5nn]) )
                        prednmed.append( sum([lab.split('_')[1]=='Regional' for lab in class5nn]) )
                        prednmed.append( sum([lab.split('_')[1]=='Segmental' for lab in class5nn]) )
                        prednmed.append( sum([lab.split('_')[1]=='N/A' for lab in class5nn]) )
                        prednmeie=[]
                        prednmeie.append( sum([lab.split('_')[2]=='Clumped' for lab in class5nn]) )
                        prednmeie.append( sum([lab.split('_')[2]=='ClusteredRing' for lab in class5nn]) )
                        prednmeie.append( sum([lab.split('_')[2]=='Heterogeneous' for lab in class5nn]) )
                        prednmeie.append( sum([lab.split('_')[2]=='Homogeneous' for lab in class5nn]) )
                        prednmeie.append( sum([lab.split('_')[2]=='Stippled or punctate' for lab in class5nn]) )                        
                        prednmeie.append( sum([lab.split('_')[2]=='N/A' for lab in class5nn]) )
                        # predicion based on majority
                        pred_nme_dist = [nme_dist[l] for l,pc in enumerate(prednmed) if pc>=max(prednmed) and max(prednmed)>0]    
                        pred_nme_intenh = [nme_intenh[l] for l,pc in enumerate(prednmeie) if pc>=max(prednmeie) and max(prednmeie)>0] 

                        # allow a second kind of detection rate - a nme similar local neighborhood
                        if(iclass.split('_')[1] in pred_nme_dist or iclass.split('_')[2] in pred_nme_intenh):
                            NME_descriptorate.append(1)
                        
                        # for nme_dist
                        label_nme_dist = [l for l,pc in enumerate(nme_dist) if iclass.split('_')[1]==pc]   
                        labelpred_nme_dist = [l for l,pc in enumerate(prednmed) if pc>=max(prednmed) and max(prednmed)>0]  
                        for u in label_nme_dist:
                            for v in labelpred_nme_dist:
                                wnme_dist[v,u] += 1
                                
                        # for nme_intenh     
                        label_nme_intenh = [l for l,pc in enumerate(nme_intenh) if iclass.split('_')[2]==pc]   
                        labelpred_intenh  = [l for l,pc in enumerate(prednmeie) if pc>=max(prednmeie) and max(prednmeie)>0]  
                        for u in label_nme_intenh:
                            for v in labelpred_intenh:
                                wnme_intenh[v,u] += 1
              
                # compute Z-space Accuracy
                Acc = scores.mean()
                print "cvRF Z-space Accuracy = %f " % Acc
                print scores.tolist()

                NME_Acc = sum(np.asarray(NME_descriptorate))/float(kznme)
                print"NME decriptor agreenment (NMErate) = %f " % NME_Acc
                indwnme_dist = linear_assignment(wnme_dist.max() - wnme_dist)
                indwnme_intenh = linear_assignment(wnme_intenh.max() - wnme_intenh)
                Acc5nn_nme_dist = sum([wnme_dist[v,u] for v,u in indwnme_dist])/float(kznme)
                Acc5nn_nme_intenh = sum([wnme_intenh[v,u] for v,u in indwnme_intenh])/float(kznme)
                print"Acc5nn_nme_dist (DIST) = %f " % Acc5nn_nme_dist
                print"Acc5nn_nme_intenh (INT_ENH) = %f " % Acc5nn_nme_intenh    
                
                if(i==0):
                    tsne = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
                                init='pca', random_state=0, verbose=2, method='exact')
                    Z_tsne = tsne.fit_transform(z)        
                    self.best_args['initAcc'] = Acc
                    # plot initial z        
                    figinint = plt.figure()
                    axinint = figinint.add_subplot(1,1,1)
                    plot_embedding_unsuper_NMEdist_intenh(Z_tsne, named_y, axinint, title='kmeans init tsne: Acc={}\n NME_Acc={}\n Acc5nn_nme_dist={}\n Acc5nn_nme_intenh={}'.format(Acc,NME_Acc,Acc5nn_nme_dist,Acc5nn_nme_intenh), legend=True)
                    figinint.savefig('{}//tsne_init_z{}_mu{}_{}.pdf'.format(save_to,self.best_args['znum'],self.best_args['num_centers'],labeltype), bbox_inches='tight')     
                    plt.close()                  
                    
                # save best args
                self.best_args['acci'].append( Acc )
                if(Acc >= self.maxAcc):
                    print 'Improving maxAcc = {}'.format(Acc)
                    for key, v in args.items():
                        self.best_args[key] = args[key]
                        
                    self.maxAcc = Acc
                    self.best_args['zbestacci']  = z 
                    self.best_args['bestacci'].append( Acc )
                    self.best_args['dec_mu'][:] = args['dec_mu'].asnumpy()
                    self.best_args['NME_Acc'] = NME_Acc
                    self.best_args['Acc5nn_nme_dist'] = Acc5nn_nme_dist
                    self.best_args['Acc5nn_nme_intenh'] = Acc5nn_nme_intenh
                
                if(i>0 and i%self.best_args['plot_interval']==0 and self.ploti<=15): 
                    # Visualize the progression of the embedded representation in a subsample of data
                    # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
                    tsne = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
                         init='pca', random_state=0, verbose=2, method='exact')
                    Z_tsne = tsne.fit_transform(z)
                    axprogress = figprogress.add_subplot(4,4,1+self.ploti)
                    plot_embedding_unsuper_NMEdist_intenh(Z_tsne, named_y, axprogress, title="Epoch %d z_tsne iter (%d)" % (self.ploti,i), legend=False)
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
                    self.best_args['p'] = p
                    self.best_args['z']  = z      
                    self.best_args['acci'].append( Acc )
                    return True 
                    
                self.best_args['y_pred'] = y_pred
                self.best_args['p'] = p
                self.best_args['z']  = z 

        # start solver
        solver.set_iter_start_callback(refresh)
        solver.set_monitor(Monitor(50))
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
        plot_embedding_unsuper_NMEdist_intenh(Z_tsne, self.best_args['named_y'], axfinal, title='final tsne: Acc={}\n NME_Acc={} \n Acc5nn_nme_dist={}\n Acc5nn_nme_intenh={}'.format(self.best_args['bestacci'][-1],self.best_args['NME_Acc'],self.best_args['Acc5nn_nme_dist'],self.best_args['Acc5nn_nme_intenh']), legend=True)
        figfinal.savefig('{}\\tsne_final_z{}_mu{}_{}.pdf'.format(save_to,self.best_args['znum'],self.best_args['num_centers'],labeltype), bbox_inches='tight')    
        plt.close()          

        outdict = {'acc': self.best_args['acci'],
                   'bestacc': self.best_args['bestacci'],
                    'NME_Acc': self.best_args['NME_Acc'],
                    'Acc5nn_nme_dist':self.best_args['Acc5nn_nme_dist'],
                    'Acc5nn_nme_intenh':self.best_args['Acc5nn_nme_intenh'],
                    'p': self.best_args['p'],
                    'z': self.best_args['z'],
                    'y_pred': self.best_args['y_pred'],
                    'named_y': self.best_args['named_y'],
                    'num_centers': self.best_args['num_centers']}
            
        return outdict
        
            
if __name__ == '__main__':
    #####################################################
    from decModel_exp4 import *
    from utilities import *
    
    ## 1) read in the datasets both all NME (to do pretraining)
    NME_nxgraphs = r'Z:\Cristina\Section3\NME_DEC\imgFeatures\NME_nxgraphs'
    # start by loading nxGdatafeatures
    with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures.pklz'), 'rb') as fin:
        nxGdatafeatures = pickle.load(fin)
        
    with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_dynamic.pklz'), 'rb') as fin:
        allNMEs_dynamic = pickle.load(fin)
        
    with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_morphology.pklz'), 'rb') as fin:
        allNMEs_morphology = pickle.load(fin)        
        
    with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_texture.pklz'), 'rb') as fin:
        allNMEs_texture = pickle.load(fin)
        
    with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_stage1.pklz'), 'rb') as fin:
        allNMEs_stage1 = pickle.load(fin) 
        
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
        
    nxGdiscfeatures = np.concatenate([gdiscf for gdiscf in normgdiscf], axis=1)
    # append other univariate features  nxGdiscfeatures.shape  (798L, 422L)               
    nxGdiscfeatures = np.concatenate((nxGdiscfeatures,                    
                                        ds.reshape(len(ds),1),
                                        ms.reshape(len(ms),1)), axis=1)
    # shape input 
    combX_allNME = np.concatenate((alldiscrSERcounts, nxGdiscfeatures, normdynamic, normorpho, normtext, normstage1), axis=1)          
    YnxG_allNME = [nxGdatafeatures['roi_id'].values,
            nxGdatafeatures['roi_label'].values,
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
    YnxG_filledbyBC = [nxGdatafeatures['roi_id'].values,
            nxGdatafeatures['roi_label'].values,
            nxGdatafeatures['roiBIRADS'].values,
            nxGdatafeatures['NME_dist'].values,
            nxGdatafeatures['NME_int_enh'].values]

    print('Loading {} NME filled by BC of size = {}'.format(combX_filledbyBC.shape[0], combX_filledbyBC.shape[1]) )
    print('Loading NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_filledbyBC[0].shape[0])   )
                                              
    ######################
    ## 2) DEC using labeled cases
    ######################                                            
    labeltype = 'wimgF_varying_mu' # roilabel_NME_dist
    save_to = r'Z:\Cristina\Section3\NME_DEC\SAEmodels\decModel_exp4_Z-spaceAcc'
   
    #log
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    LOG_FILE_stdout = os.path.join(save_to, 'allDEC_batch_exp4_{}.txt'.format(labeltype))
    #log to console    
    console_formatter = logging.Formatter('%(asctime)s %(message)s')
    #ch = logging.StreamHandler()
    ch = logging.FileHandler(LOG_FILE_stdout,'a')
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)
        
    # dfine num_centers according to clustering variable
    ## use y_dec to  minimizing KL divergence for clustering with known classes
    X = combX_allNME  
    ysup = YnxG_allNME[1]+'_'+YnxG_allNME[3]+'_'+YnxG_allNME[4]
    ysup[range(combX_filledbyBC.shape[0])] = YnxG_filledbyBC[1]+'_'+YnxG_filledbyBC[3]+'_'+YnxG_filledbyBC[4] # +['_'+str(yl) for yl in YnxG_filledbyBC[3]]  
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
    # DEC w varying_mu
    ########################################################
    input_size = combX_allNME.shape[1]
    latent_size = [input_size/rxf for rxf in [5]] #25,15,10,
    varying_mu = [int(np.round(var_mu)) for var_mu in np.linspace(8,20,13)] # 3,20,18
    alphaT = [10*(2**j) for j in range(9)]
    print alphaT
    
    # to load a prevously DEC model
    for znum in latent_size:
        dfDEC_varmu_perf = pd.DataFrame()
        normalizedMI = []
        maxAccuracy = []        
        cvRFAccuracy = []
        initAccuracy = []
        NME_Acc = []
        NME_DISTAcc = []
        NME_INTENHAcc = []
        cvRFNME_DISTAcc = []
        cvRFNME_INTENHAcc = []
        for muk,num_centers in enumerate(varying_mu):

            batch_size = 25
            # Read autoencoder: note is not dependent on number of clusters just on z latent size
            print "Load autoencoder of znum = ",znum
            print "Training DEC num_centers = ",num_centers
            logger.info('Load autoencoder of znum = {}\n Training DEC num_centers ={}'.format(znum,num_centers))
            dec_model = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 
            logger.info('Tunning DEC batch_size ={}, alpha anheling={}'.format(batch_size,alphaT[2])) # orig paper 256*40 (10240) point for upgrade about 1/6 (N) of data
            outdict = dec_model.cluster_varying_mu(X, y_dec, roi_labels, classes, batch_size, save_to, labeltype, update_interval=40) # 10 epochs# ~ per 1/3 of data 798/48=16 update twice per epoch ~ N/(batch size)=iterations to reach a full epochg
            logger.info('Finised trainining DEC...') 
            print dec_model.best_args['bestacci']
            logger.info('dec_model.best_args[bestacci]={}'.format(dec_model.best_args['bestacci']))
            initAccuracy.append( dec_model.best_args['initAcc'] )
            outdict['initAcc'] = dec_model.best_args['initAcc']
            # find NME descriptor agreement in best
            NME_Acc.append( dec_model.best_args['NME_Acc'] )
            outdict['NME_Acc'] = dec_model.best_args['NME_Acc']
            # find unsupervised accuracy of NME
            NME_DISTAcc.append(dec_model.best_args['Acc5nn_nme_dist'])
            NME_INTENHAcc.append(dec_model.best_args['Acc5nn_nme_intenh'])
            outdict['NME_DISTAcc'] = dec_model.best_args['Acc5nn_nme_dist']
            outdict['NME_INTENHAcc'] = dec_model.best_args['Acc5nn_nme_intenh']
            
            # save output results
            dec_args_keys = ['encoder_1_bias', 'encoder_3_weight', 'encoder_0_weight', 
            'encoder_0_bias', 'encoder_2_weight', 'encoder_1_weight', 
            'encoder_3_bias', 'encoder_2_bias']
            dec_args = {key: v.asnumpy() for key, v in dec_model.best_args.items() if key in dec_args_keys}
            dec_args['dec_mubestacci'] = dec_model.best_args['dec_mu']
            args_save = {key: v for key, v in dec_model.best_args.items() if key not in dec_args_keys}
            dec_model = dec_args.copy()
            dec_model.update(args_save) 
            with open(fname, 'w') as fout:
                pickle.dump(args_save, fout)
            # save model saving params into a numpy array
            dec_model_save= gzip.open(os.path.join(save_to,'dec_model_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'wb')
            pickle.dump(dec_model, dec_model_save, protocol=pickle.HIGHEST_PROTOCOL)
            dec_model_save.close()
            
            ## plot iterations
            df = pd.DataFrame({'acc': pd.Series(outdict['acc']), 'iterations':range(len(outdict['acc'])), 'num_centers':outdict['num_centers']})   
            fig2 = plt.figure(figsize=(20,6))
            ax2 = plt.axes()
            sns.set_context("notebook")
            sns.set_style("darkgrid", {"axes.facecolor": ".9"})    
            sns.pointplot(x="iterations", y="acc", hue="num_centers", data=df, ax=ax2, size=0.05) 
            fig2.autofmt_xdate(bottom=0.2, rotation=30, ha='right')   
            fig2.savefig(save_to+os.sep+'DEC_z{}_mu{}_{}-unsuprv acc vs iteration.pdf'.format(znum,num_centers,labeltype), bbox_inches='tight')    
            plt.close(fig2)
                    
            #####################
            # algorithm to find the epoch at best acc 
            #####################
            max_acc = 0
            epochs = []; best_acc = []
            for k,acc in enumerate(dec_model['acci']):
                if(acc >= max_acc):
                    epochs.append(k)
                    best_acc.append(acc)
                    max_acc = acc
            # to plot best acci
            dfDEC_varmu_perf = dfDEC_varmu_perf.append( pd.DataFrame({'bestacci': pd.Series(best_acc), 'iterations':epochs, 'num_centers':num_centers, 'znum':znum }) )
            maxAccuracy.append(max_acc)
            outdict['max_acc'] = max_acc
            logger.info('dec_model max_acc={}'.format(max_acc))

            #####################
            # Calculate normalized MI: find the relative frequency of points in Wk and Cj
            #####################
            N = X.shape[0]
            num_classes = len(np.unique(roi_labels)) # present but not needed during AE training
            roi_classes = np.unique(roi_labels)
            roi_labels = np.asarray(roi_labels)
            
            # to get final cluster memberships
            # extact best params
            # extact embedding space
            all_iter = mx.io.NDArrayIter({'data': X}, batch_size=X.shape[0], shuffle=False,
                                                      last_batch_handle='pad')   
            ## embedded point zi 
            aDEC = DECModel(mx.cpu(), X, num_centers, 1.0, znum, 'Z:\\Cristina\\Section3\\NME_DEC\\SAEmodels') 
            mxdec_args = {key: mx.nd.array(v) for key, v in dec_args.items() if key != 'dec_mubestacci'}                           
            zbestacci = model.extract_feature(aDEC.feature, mxdec_args, None, all_iter, X.shape[0], aDEC.xpu).values()[0]      
            # orig paper 256*40 (10240) point for upgrade about 1/6 (N) of data
            #zbestacci = dec_model['zbestacci'] 
            pbestacci = np.zeros((zbestacci.shape[0], dec_model['num_centers']))
            aDEC.dec_op.forward([zbestacci, dec_args['dec_mubestacci'].asnumpy()], [pbestacci])
            y = np.asarray(roi_labels)
    
            # find max soft assignments
            W = pbestacci.argmax(axis=1)
            num_clusters = len(np.unique(W))
            clusters = np.unique(W)
            
            MLE_kj = np.zeros((num_clusters,num_classes))
            absWk = np.zeros((num_clusters))
            absCj = np.zeros((num_classes))
            for k in range(num_clusters):
                # find poinst in cluster k
                absWk[k] = np.sum(W==clusters[k])
                for j in range(num_classes):
                    # find points of class j
                    absCj[j] = np.sum(roi_labels==roi_classes[j])
                    # find intersection 
                    ptsk = W==clusters[k] 
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
            Hw = -np.sum(absWk/N*np.log(absWk/N))
            NMI = Iwk/((Hc+Hw)/2.0)
            print "... DEC normalizedMI = ", NMI
            # to plot best acci
            normalizedMI.append( NMI ) 
            outdict['NMI'] = NMI
            logger.info('dec_model NMI={}'.format(NMI))            
            
            # save to R
            pdzfinal =  pd.DataFrame({'roilabels': YnxG_allNME[1], 'NME_dist': YnxG_allNME[3], 'NME_int_enh': YnxG_allNME[4]})
            dfzbestacci =  pd.DataFrame({zvar: zbestacci[:,zvar] for zvar in range(zbestacci.shape[1])})
            zfinal = pd.concat([pdzfinal, dfzbestacci], axis=1)
            zfinal.to_csv(save_to+os.sep+'zfinal_z{}_mu{}_{}.csv'.format(znum,num_centers,labeltype), sep=',', encoding='utf-8', header=False, index=False)
            
            ######## train an RF on ORIGINAL space Finally perform a cross-validation using RF
            datalabels = YnxG_allNME[1]
            RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=200, random_state=0, verbose=1)
            # Evaluate a score by cross-validation
            # integer=5, to specify the number of folds in a (Stratified)KFold,
            scores = cross_val_score(RFmodel, X, datalabels, cv=5)
            print "(cvRF Accuracy = %f " % scores.mean()      
            print scores.tolist()
            cvRFAccuracy.append(scores.mean())
            outdict['cvRFAccuracy'] = scores.mean() 
            logger.info('all cv OriginalX RF_Accuracy ={}'.format( scores.tolist() ))
            
            ######## train an RF on ORIGINAL space Finally perform a cross-validation using RF
            data = X[YnxG_allNME[3]!='N/A',:]
            datalabels = YnxG_allNME[3][YnxG_allNME[3]!='N/A']
            RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=200, random_state=0, verbose=1)
            # Evaluate a score by cross-validation
            # integer=5, to specify the number of folds in a (Stratified)KFold,
            scores = cross_val_score(RFmodel, data, datalabels, cv=5)
            print "(cvRF NME_DISTAcc_Accuracy  = %f " % scores.mean()      
            print scores.tolist()
            cvRFNME_DISTAcc.append(scores.mean())
            outdict['cvRFNME_DISTAcc'] = scores.mean() 
            logger.info('all cv RF NME_DISTAcc_Accuracy ={}'.format( scores.tolist() ))
            
            ######## train an RF on ORIGINAL space Finally perform a cross-validation using RF
            data = X[YnxG_allNME[4]!='N/A',:]
            datalabels = YnxG_allNME[4][YnxG_allNME[4]!='N/A']
            RFmodel = RandomForestClassifier(n_jobs=2, n_estimators=200, random_state=0, verbose=1)
            # Evaluate a score by cross-validation
            # integer=5, to specify the number of folds in a (Stratified)KFold,
            scores = cross_val_score(RFmodel, data, datalabels, cv=5)
            print "(cvRF NME_INTENH Accuracy = %f " % scores.mean()      
            print scores.tolist()
            cvRFNME_INTENHAcc.append(scores.mean())
            outdict['cvRFNME_INTENHAcc'] = scores.mean() 
            logger.info('all cv RF NME_INTENH Accuracy ={}'.format( scores.tolist() ))
            
            # save model saving params into a numpy array
            outdict_save= gzip.open(os.path.join(save_to,'outdict_z{}_mu{}_{}.arg'.format(znum,num_centers,labeltype)), 'wb')
            pickle.dump(outdict, outdict_save, protocol=pickle.HIGHEST_PROTOCOL)
            outdict_save.close()

        ##### finish plotting by each znum
        # plot latent space Accuracies vs. original
        colors = plt.cm.jet(np.linspace(0, 1, 16))
        fig1 = plt.figure(figsize=(20,6))
        ax1 = plt.axes()
        ax1.plot(varying_mu, initAccuracy, color=colors[0])
        ax1.plot(varying_mu, maxAccuracy, color=colors[2])
        ax1.plot(varying_mu, cvRFAccuracy, color=colors[4], ls='-.')  ## Average malignant and  benigng classs
        ax1.plot(varying_mu, NME_Acc, color=colors[6], ls='-')
        ax1.plot(varying_mu, NME_DISTAcc, color=colors[8], ls='-')
        ax1.plot(varying_mu, NME_INTENHAcc, color=colors[10], ls='-')
        ax1.plot(varying_mu, cvRFNME_DISTAcc, color=colors[12], ls='-.') 
        ax1.plot(varying_mu, cvRFNME_INTENHAcc, color=colors[14], ls='-.') 
        c_patchs = []
        c_patchs.append(mpatches.Patch(color=colors[0], label='initAccuracy'))
        c_patchs.append(mpatches.Patch(color=colors[2], label='max_cvRF_Zspace'))
        c_patchs.append(mpatches.Patch(color=colors[4], label='OriginalX_Acc_Malignant&Benign')) 
        c_patchs.append(mpatches.Patch(color=colors[6], label='NME_Acc'))
        c_patchs.append(mpatches.Patch(color=colors[8], label='Acc5nn_nme_dist'))        
        c_patchs.append(mpatches.Patch(color=colors[10], label='Acc5nn_nme_intenh'))    
        c_patchs.append(mpatches.Patch(color=colors[12], label='OriginalX_Acc_NME_DIST'))              
        c_patchs.append(mpatches.Patch(color=colors[14], label='OriginalX_Acc_INTENH'))  
        plt.legend(handles=c_patchs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':10})
        fig1.savefig(save_to+os.sep+'allDEC_varying_mu_Accu_NMI_z{}_{}-unsuprv acc vs iteration.pdf'.format(znum,labeltype), bbox_inches='tight')    


     # save to R
#    pdzfinal = pd.DataFrame( np.append( y[...,None], zfinal, 1) )
#    pdzfinal.to_csv('datasets//zfinal.csv', sep=',', encoding='utf-8', header=False, index=False)
#    # to save to csv
#    pdcombX = pd.DataFrame( np.append( y[...,None], combX, 1) )
#    pdcombX.to_csv('datasets//combX.csv', sep=',', encoding='utf-8', header=False, index=False)
#        


