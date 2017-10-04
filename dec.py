# pylint: skip-file
import sys
import os
import mxnet as mx
import numpy as np
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
            self.mask *= (self.alpha+1.0)/self.alpha*(p-q)
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
        self.sep = int(X.shape[0]*0.75)
        X_train = X[:self.sep]
        X_val = X[self.sep:]
        batch_size = 32 # 160 32*5 = update_interval*5
        # Train or Read autoencoder: note is not dependent on number of clusters just on z latent size
        ae_model = AutoEncoderModel(self.xpu, [X.shape[1],500,500,2000,znum], pt_dropout=0.2)
        if not os.path.exists(save_to+'_pt.arg'):
            ae_model.layerwise_pretrain(X_train, batch_size, 50000, 'sgd', l_rate=0.1, decay=0.0,
                                        lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
            ae_model.finetune(X_train, batch_size, 100000, 'sgd', l_rate=0.1, decay=0.0,
                              lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
            ae_model.save(save_to+'_pt.arg')
            logging.log(logging.INFO, "Autoencoder Training error: %f"%ae_model.eval(X_train))
            logging.log(logging.INFO, "Autoencoder Validation error: %f"%ae_model.eval(X_val))
        else:
            ae_model.load(save_to+'_pt.arg')
            logging.log(logging.INFO, "Reading Autoencoder from file..: %s"%(save_to+'_pt.arg'))
            logging.log(logging.INFO, "Autoencoder Training error: %f"%ae_model.eval(X_train))
            logging.log(logging.INFO, "Autoencoder Validation error: %f"%ae_model.eval(X_val))
            
        self.ae_model = ae_model
        logging.log(logging.INFO, "finished reading Autoencoder from file..: ")
        # prep model for clustering
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
        self.G = self.ae_model.eval(X_train)/self.ae_model.eval(X_val)


    def cluster(self, X, y, classes, fighome, update_interval=None):
        N = X.shape[0]
        if not update_interval:
            update_interval = int(self.batch_size/5.0)
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
        pp = 15
        tsne = TSNE(n_components=2, perplexity=pp, learning_rate=275,
             init='pca', random_state=0, verbose=2, method='exact')
        Z_tsne = tsne.fit_transform(z)
        
        # plot initial z        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        named_y = [classes[kc] for kc in y]
        plot_embedding(Z_tsne, named_y, ax, title="tsne with perplexity %d" % pp, legend=True, plotcolor=True)
        fig.savefig(fighome+os.sep+'tsne_init_k'+str(self.num_centers)+'_z'+str(self.znum)+'.pdf', bbox_inches='tight')    
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
        solver = Solver('sgd', momentum=0.9, wd=0.0, learning_rate=0.01)
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
        print 'update_interval = %f'%  update_interval
        
        def refresh(i):
            if i%update_interval == 0:
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
                                             
                if(i%self.batch_size==0 and self.ploti<=15): 
                    # Visualize the progression of the embedded representation in a subsample of data
                    # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
                    tsne = TSNE(n_components=2, perplexity=15, learning_rate=275,
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
                if i == self.batch_size*20: # performs 1epoch = 615/3 = 205*1000epochs #np.sum(y_pred != self.y_pred) < 0.001*y_pred.shape[0]:                    
                    self.y_pred = y_pred
                    return True 
                    
                self.y_pred = y_pred
                self.p = p
                self.z = z

        # start solver
        solver.set_iter_start_callback(refresh)
        solver.set_monitor(Monitor(50))

        solver.solve(self.xpu, self.loss, args, self.args_grad, None,
                     train_iter, 0, 1000000000, {}, False)
        self.end_args = args
        
        outdict = {'acc': self.acci,
                   'p': self.p,
                   'z': self.z,
                   'y_pred': self.y_pred,
                   'named_y': named_y}
            
        return outdict


    def cluster_unsuperv(self, X, y, y_tsne, fighome, update_interval=None):
        N = X.shape[0]
        plotting_interval = N
        if not update_interval:
            update_interval = int(self.batch_size/5.0)

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
        pp = 15
        tsne = TSNE(n_components=2, perplexity=pp, learning_rate=275,
             init='pca', random_state=0, verbose=2, method='exact')
        Z_tsne = tsne.fit_transform(z)
        
        # plot initial z        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plot_embedding(Z_tsne, y_tsne, ax, title="tsne with perplexity %d" % pp, legend=True, plotcolor=True)
        fig.savefig(fighome+os.sep+'tsne_init_z'+str(self.znum)+'.pdf', bbox_inches='tight')    
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
        solver = Solver('sgd', momentum=0.9, wd=0.0, learning_rate=0.01)
        def ce(label, pred):
            return np.sum(label*np.log(label/(pred+0.000001)))/label.shape[0]
        solver.set_metric(mx.metric.CustomMetric(ce))

        label_buff = np.zeros((X.shape[0], self.num_centers))
        train_iter = mx.io.NDArrayIter({'data': X}, {'label': label_buff}, batch_size=batch_size,
                                       shuffle=False, last_batch_handle='roll_over')
        self.y_pred = np.zeros((X.shape[0]))
        self.solvermetric = []
        self.ploti = 0
        fig = plt.figure(figsize=(20, 15))
        print 'Batch_size = %f'% self.batch_size
        print 'update_interval = %f'%  update_interval
        print 'tolernace = len(ypred)/1000 = %f'% float(0.001*self.y_pred.shape[0])
        
        def refresh_unsuperv(i):
            if i%update_interval == 0:
                print '... Updating i = %f' % i 
                z = model.extract_feature(self.feature, args, None, test_iter, N, self.xpu).values()[0]
                p = np.zeros((z.shape[0], self.num_centers))
                self.dec_op.forward([z, args['dec_mu'].asnumpy()], [p])
                # the soft assignments qi (pred)
                y_pred = p.argmax(axis=1)
                print np.std(np.bincount(y_pred)), np.bincount(y_pred)

                if y is not None:
                    # compare soft assignments with known labels (unused)
                    print np.std(np.bincount(y.astype(np.int))), np.bincount(y.astype(np.int))
                    print y_pred[0:5], y.astype(np.int)[0:5]    
                    print 'Clustering Acc = %f'% cluster_acc(y_pred, y)[0]
                    self.acci.append( cluster_acc(y_pred, y)[0] )
                    
                ## COMPUTING target distributions P
                ## we compute pi by first raising qi to the second power and then normalizing by frequency per cluster:
                weight = 1.0/p.sum(axis=0) # p.sum provides fj
                weight *= self.num_centers/weight.sum()
                p = (p**2)*weight
                train_iter.data_list[1][:] = (p.T/p.sum(axis=1)).T
                print "sum(I(y'-1!=y) = %f" % np.sum(y_pred != self.y_pred)
                self.solvermetric.append( solver.metric.get()[1] )
                print "solver.metric = %f" % solver.metric.get()[1]                
                
                # For the purpose of discovering cluster assignments, we stop our procedure when less than tol% of points change cluster assignment between two consecutive iterations.
                # tol% = 0.001
#                if np.sum(y_pred != self.y_pred) < 0.001*y_pred.shape[0]: # performs 1epoch = 615/3 = 205*1000epochs #                    
#                    self.y_pred = y_pred
#                    return True 
                    
                self.y_pred = y_pred
                self.p = p
                self.z = z
            
            # to plot
            if i%plotting_interval == 0:                              
                if(self.ploti<=15): 
                    # Visualize the progression of the embedded representation in a subsample of data
                    # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
                    tsne = TSNE(n_components=2, perplexity=15, learning_rate=275,
                         init='pca', random_state=0, verbose=2, method='exact')
                    Z_tsne = tsne.fit_transform(self.z)
                    
                    ax = fig.add_subplot(3,4,1+self.ploti)
                    plot_embedding(Z_tsne, y_tsne, ax, title="Epoch %d z_tsne iter (%d)" % (self.ploti,i), legend=False, plotcolor=True)
                    self.ploti = self.ploti+1

        # start solver
        solver.set_iter_start_callback(refresh_unsuperv)
        # monitor every self.batch_size
        solver.set_monitor(Monitor(self.batch_size))
        solver.solve(self.xpu, self.loss, args, self.args_grad, None,
                     train_iter, 0, 12*N, {}, False)
        # finish                
        fig = plt.gcf()
        fig.savefig(fighome+os.sep+'tsne_progress_k'+str(self.num_centers)+'_z'+str(self.znum)+'.pdf', bbox_inches='tight')    
        plt.close()          
        
        # plot progression of clustering loss
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(range(len(self.solvermetric)),self.solvermetric, '-.')  
        ax.set_xlabel("iter")
        ax.set_ylabel("L loss for num_centers ="+str(self.num_centers))
        fig.savefig(fighome+os.sep+'clustering_loss_numcenters'+str(self.num_centers)+'_z'+str(self.znum)+'.pdf', bbox_inches='tight')    
        plt.close()
            
        self.end_args = args        
        outdict = {'p': self.p,
                   'z': self.z,
                   'y_pred': self.y_pred}
            
        return outdict        
        
    def tsne_wtest(self, X_test, y_wtest, zfinal):
        ## self=dec_model
        ## embedded point zi and
        X_test = combX[0,:]
        X_test = np.reshape(X_test, (1,X_test.shape[0]))
        y_test = y[0]
        y_wtest = list(y)
        y_wtest.append(y_test)
        y_wtest = np.asarray(y_wtest)
        # X_wtest = np.vstack([combX, X_test])
               
        ########
        fighome = 'results//clusterDEC_unsuperv_QuantitativEval_wPerfm'
        znum = 10
        numc = 19    
        dec_model = DECModel(mx.cpu(), combX, numc, 1.0, znum, 'model/NME_'+str(znum)+'k')
        with open(fighome+os.sep+'NME_Quantitative_NMI_numc'+str(numc)+'_'+str(znum)+'z', 'rb') as fin:
           savedict = pickle.load(fin)
        
        
        N = combX.shape[0] #X_test.transpose().shape
        test_iter = mx.io.NDArrayIter({'data': combX}, 
                                      batch_size=1, 
                                      shuffle=False,
                                      last_batch_handle='pad')
        args = {k: mx.nd.array(v.asnumpy(), ctx=dec_model.xpu) for k, v in savedict['end_args'].items()}
        #  dec_for
        z_test = model.extract_feature(dec_model.feature, args, None, test_iter, N, dec_model.xpu).values()[0]
               
        # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
        tsne = TSNE(n_components=2, perplexity=15, learning_rate=375,
             init='pca', random_state=0, verbose=2, method='exact')
        Z_tsne = tsne.fit_transform(z_test)
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        y_tsne = savedict['clusterpts_labels']
        plot_embedding(Z_tsne, y_tsne, ax, title="tsne with test y class(%s)" % (y_tsne[-1]), legend=True, plotcolor=True)
        
        # for cluster prediction 
        p_test = np.zeros((z_test.shape[0], dec_model.num_centers))
        dec_mu_final = args['dec_mu'].asnumpy()
        dec_model.dec_op.forward([z_test, dec_mu_final], [p_test])
        # the soft assignments qi (pred)
        y_test_pred = p_test.argmax(axis=1)
        print y_test_pred
        
        
if __name__ == '__main__':    
    ########################################################    
    ## BIRADS model
    ########################################################
    from dec import *
    from utilities import *
    logging.basicConfig(level=logging.INFO)
    
    # 1. Read datasets
    Xserw, Yserw = data.get_serw()
    XnxG, YnxG, nxGdata = data.get_nxGfeatures()
    # 2. combining normSERcounts + nxGdatafeatures
    # total variable 24+455
    combX = np.concatenate((Xserw,XnxG), axis=1)
    print(combX.shape)
    # preparation for fig 5.
    from utilities import make_graph_ndMRIdata
    imgd = []
    for roi_id in range(1,len(nxGdata)+1):   
        imgd.append( make_graph_ndMRIdata(roi_id, typenxg='MST'))
    ## use y_dec to  minimizing KL divergence for clustering with known classes
    y = np.asarray(['K' if la=='U' else la for la in YnxG[1]]).astype(object)
    y = y+['_'+str(yl) for yl in YnxG[3]] #YnxG[0]+YnxG[1]
    try:
        y_dec = np.asarray([int(label) for label in y])
    except:
        classes = [str(c) for c in np.unique(y)]
        numclasses = [i for i in range(len(classes))]
        ynum_dec = y
        y_dec = []
        for k in range(len(y)):
            for j in range(len(classes)):
                if(str(y[k])==classes[j]): 
                    y_dec.append(numclasses[j])
        y_dec = np.asarray(y_dec)
    
    ########################################################
    # To build/Read AE
    ########################################################
    y = YnxG[0] #+YnxG[1]
    num_centers = len(np.unique(y)) # present but not needed during AE training
    ae_znum = [10,15,20,30,40,50]
    for znum in ae_znum:
        print "Building autoencoder of latent size znum = ",znum
        dec_model = DECModel(mx.cpu(), combX, num_centers, 1.0, znum, 'model/NME_'+str(znum)+'k')
    
    # Plot fig 5: Gradient vs. soft assigments before KL divergence
    tmm = TMM(n_components=num_centers, alpha=1.0)
    tmm.fit(combX)
    vis_gradient_NME(combX, tmm, imgd, nxGdata, titleplot='results//vis_gradient_BIRADS_k'+str(num_centers)+'_z'+str(znum) )
        
    ########################################################
    ## Quanlitative evaluation        
    ## cluster DEC for BIRADS score prediction
    ########################################################
    fighome = 'results//clusterDEC_QuanlitativEval'
    ## visualize the progression of the embedded representation of a random subset data 
    # during training. For visualization we use t-SNE (van der Maaten & Hinton, 2008) 
    # applied to the embedded points z
    num_centers = len(np.unique(y)) # present but not needed during AE training
    ae_znum = [10,15,20,30,40,50]
    clusteringAc_znum = []
    for znum in ae_znum:
        print "Runing DEC with autoencoder of latent size znum = ",znum
        dec_model = DECModel(mx.cpu(), combX, num_centers, 1.0, znum, 'model/NME_'+str(znum)+'k')
        outdict = dec_model.cluster(combX, y_dec, classes, fighome, update_interval=32)
        # save
        with open('model/NME_clusters_'+str(znum)+'k', 'wb') as fout:
            pickle.dump(outdict, fout)
        # to load  
        #with open('model/NME_clusters_'+str(znum)+'k','rb') as fin:
        #    outdict = pickle.load(fin)
            
        fig = plt.gcf()
        fig.savefig(fighome+os.sep+'tsne_progress_k'+str(num_centers)+'_z'+str(znum)+'.pdf', bbox_inches='tight')    
        plt.close()    
    
        pfinal = outdict['p']
        zfinal = outdict['z']
        clusteringAc_znum.append( outdict['acc'][-1] )
        # to get final cluster memberships
        ypredfinal = pfinal.argmax(axis=1)
        
        # plot final z  
        tsne = TSNE(n_components=2, perplexity=15, learning_rate=75, init='pca', random_state=0, verbose=2, method='exact')
        Z_tsne = tsne.fit_transform(zfinal)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plot_embedding(Z_tsne, y, ax, title="tsne embedding space after DEC", legend=True, plotcolor=True)
        fig.savefig(fighome+os.sep+'tsne_final_k'+str(num_centers)+'_z'+str(znum)+'.pdf', bbox_inches='tight')    
        plt.close() 
        
        # to plot top scoring cluster members
        vis_topscoring_NME(combX, imgd, num_centers, nxGdata, pfinal, zfinal, titleplot=fighome+os.sep+'vis_topscoring_k'+str(num_centers)+'_z'+str(znum))
        
    
    ########################################################    
    ## Experiment #1: Using different latent space dimensions: ACC of clustering
    ########################################################
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(ae_znum, clusteringAc_znum,'*-r')   
    ax.set_xlabel("size z latent space (DR vars)")
    ax.set_ylabel("Clustering labels+NME distr (num_centers="+str(num_centers)+") accuracy")
    fig.savefig(fighome+os.sep+'Clustering_acc_vs_numcenters'+str(num_centers)+'_z'+str(znum)+'.pdf', bbox_inches='tight')    
    plt.close()
    
    ########################################################
    # to illustrate: we can appreciate changes in the neighborhood of a given case, and the resulting classification based on nearest neighbors
    # Algorithm for finding Nearest Neighbors for a given tsne_id:
    #    Build a spatial.cKDTree(X_tsne, compact_nodes=True) with embedded points
    #    1) start with a neighborhood radius set of 0.05 or 5% in map space
    #    2) if a local neighboorhood is found closest to the tsne_id, append neighbors to NN_embedding_indx_list
    #        if not found, increment neighborhood radius by 1% and query neighboorhood until a local neighboorhood is found
    #                
    # plot TSNE with upto 6 nearest neighbors 
    from utilities import visualize_Zlatent_NN_fortsne_id, plot_pngs_showNN
    all6M_tsne_id = [237,238,324,329,374,386,387,579,587,589,598,599,606,607,608,609]   ## for 6M
    y_tsne =  y
    for tsne_id in all6M_tsne_id:
        pdNN = visualize_Zlatent_NN_fortsne_id(Z_tsne, y_tsne, tsne_id, saveFigs=True)

    ########################################################
    ## Quantitative evaluation        
    ## DEC for unknown number clusters
    ########################################################
    import sklearn.neighbors 
    from utilities import visualize_Zlatent_NN_fortsne_id
    import matplotlib.patches as mpatches
        
    fighome = 'results//clusterDEC_unsuperv_QuantitativEval_wPerfm'
    ## visualize the progression of the embedded representation of a random subset data 
    # during training. For visualization we use t-SNE (van der Maaten & Hinton, 2008) 
    # applied to the embedded points z
    allnum_centers = [n for n in range(3,21)] # unsupervised

    # get labels for real classes
    clusterpts_labels, clusterpts_diagnosis = data.get_pathologyLabels(YnxG)
    clusterpts_labelsorig = clusterpts_labels
    clusterpts_labels = np.asarray(clusterpts_labels, dtype='object')
    
    # implement generalizability and NMI
    generalizability = []
    normalizedMI = []
    numc_TPR = []
    numc_TNR = []
    numc_Accu = []
    znum = 20
    # to plut progress
    upto_numc = []
    figprog = plt.figure()
    axprog = figprog.add_subplot(1,1,1)
    
    for numc in allnum_centers:
        print "Runing DEC with autoencoder of num_centers = ",numc
        print "Runing DEC with autoencoder of znum = ",znum
        dec_model = DECModel(mx.cpu(), combX, numc, 1.0, znum, 'model/NME_'+str(znum)+'k')
        # cluster per each numc
        outdict = dec_model.cluster_unsuperv(combX, None, clusterpts_labels, fighome, update_interval=18)
        # save
        with open('model/NME_clusters_unsuperv_'+str(numc)+'k_'+str(znum)+'z', 'wb') as fout:
            pickle.dump(outdict, fout)
        # to load  
        #with open('model/NME_clusters_unsuperv_'+str(numc)+'k_'+str(znum)+'z','rb') as fin:
        #   outdict = pickle.load(fin)
        pfinal = outdict['p']
        zfinal = outdict['z']
        y = YnxG[1]
        
        ##########################
        # Calculate 5-nn TPR and TRN among pathological lesions
        # Sensitivity (also called the true positive rate, the recall, or probability of detection in some fields) 
        # measures the proportion of positives that are correctly identified as such (i.e. the percentage of sick people who are correctly identified as having the condition).
        # Specificity (also called the true negative rate) measures the proportion of negatives that are correctly identified as such 
        # create sklearn.neighbors
        Z_embedding_tree = sklearn.neighbors.BallTree(zfinal, leaf_size=5)     
        # This finds the indices of 5 closest neighbors
        N = sum(y==np.unique(y)[0]) #for B
        P = sum(y==np.unique(y)[1]) #for M
        TP = []
        TN = []
        for k in range(zfinal.shape[0]):
            iclass = y[k]
            dist, ind = Z_embedding_tree.query([zfinal[k]], k=6)
            dist5nn, ind5nn = dist[k!=ind], ind[k!=ind]
            class5nn = y[ind5nn]
            cluster5nn = clusterpts_labels[ind5nn]
            # exlcude U class
            class5nn = class5nn[class5nn!='U']
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
        
        # visualize and interesting case        
        tsne = TSNE(n_components=2, perplexity=15, learning_rate=75, init='pca', random_state=0, verbose=2, method='exact')
        Z_tsne = tsne.fit_transform(zfinal)       
        pdNN = visualize_Zlatent_NN_fortsne_id(Z_tsne, clusterpts_labels, 464, saveFigs=True)
        print(pdNN)
        
        ##########################                
        # Calculate normalized MI:
        # find the relative frequency of points in Wk and Cj
        N = combX.shape[0]
        num_classes = len(np.unique(clusterpts_labels)) # present but not needed during AE training
        classes = np.unique(clusterpts_labels)
        # to get final cluster memberships
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
                absCj[j] = sum(clusterpts_labels==classes[j])
                # find intersection 
                ptsk = W==k 
                MLE_kj[k,j] = sum(ptsk[clusterpts_labels==classes[j]])
        # if not assignment incluster
        absWk[absWk==0]=0.00001
        
        # compute NMI
        numIwc = np.zeros((num_clusters,num_classes))
        for k in range(num_clusters):
            for j in range(num_classes):
                if(MLE_kj[k,j]!=0):
                    numIwc[k,j] = MLE_kj[k,j]/N * np.log( N*MLE_kj[k,j]/(absWk[k]*absCj[j]) )
                
        Iwk = sum(sum(numIwc, axis=1), axis=0)       
        Hc = -sum(absCj/N*np.log(absCj/N))
        Hw = -sum(absWk/N*np.log(absWk/N))
        NMI = Iwk/((Hc+Hw)/2)
        normalizedMI.append( NMI ) 
        print "... DEC normalizedMI = ", NMI

        ##########################  
        # Calculate Loss Gradient KL divergence
        dec_mu = dec_model.end_args['dec_mu'].asnumpy()
        G = vis_absgradient_final(pfinal, zfinal, dec_mu, dec_model.sep) 
        generalizability.append( G )
        print "... DEC generalizability = ", G
        
        ##########################
        # plot progress:
        c_patchs = []
        colors=['c','b','g','r','m']
        labels=['NMI','generalizability','TPR','TNR','Accuracy']
        for k in range(5):
            c_patchs.append(mpatches.Patch(color=colors[k], label=labels[k]))
        plt.legend(handles=c_patchs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':10})
        
        upto_numc.append(numc)
        axprog.plot(upto_numc, [10*NMI for NMI in normalizedMI],'.-c')   
        axprog.plot(upto_numc, generalizability,'.-b') 
        axprog.plot(upto_numc, numc_TPR,'.-g') 
        axprog.plot(upto_numc, numc_TNR,'.-r')
        axprog.plot(upto_numc, numc_Accu,'.-m')
        axprog.set_xlabel("# clusters")
        figprog.savefig(fighome+os.sep+'NMI_generalization_num_clusters'+str(num_clusters)+'_z'+str(znum)+'.pdf', bbox_inches='tight')    
        plt.close()
        
        # save
        savedict = {'end_args': dec_model.end_args,
                    'clusterpts_labels': clusterpts_labels,
                    'NMI': NMI,
                    'G': G,
                    'TPR': TPR,
                    'TNR': TNR,
                    'Accu': Accu}
        
        with open(fighome+os.sep+'NME_Quantitative_NMI_numc'+str(numc)+'_'+str(znum)+'z', 'wb') as fout:
            pickle.dump(savedict, fout)
        
    #############
    # plot final
    figprog = plt.figure()
    axprog = figprog.add_subplot(1,1,1)
    c_patchs = []
    colors=['c','b','g','r','m']
    labels=['NMI','generalizability','TPR','TNR','Accuracy']
    for k in range(5):
        c_patchs.append(mpatches.Patch(color=colors[k], label=labels[k]))
    plt.legend(handles=c_patchs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':10})
    
    axprog.plot(allnum_centers, [10*NMI for NMI in normalizedMI],'.-c')   
    axprog.plot(allnum_centers, generalizability,'.-b') 
    axprog.plot(allnum_centers, numc_TPR,'.-g') 
    axprog.plot(allnum_centers, numc_TNR,'.-r')
    axprog.plot(allnum_centers, numc_Accu,'.-m')
    axprog.set_xlabel("# clusters")
    figprog.savefig(fighome+os.sep+'NMI_generalization_overall_z'+str(znum)+'.pdf', bbox_inches='tight')    
    
    # save
    savedict = {'end_args': dec_model.end_args,
                'clusterpts_labels': clusterpts_labels,
                'NMI': normalizedMI,
                'G': generalizability,
                'TPR': numc_TPR,
                'TNR': numc_TNR,
                'Accu':numc_Accu}
    
    with open(fighome+os.sep+'NME_Quantitative_NMI_'+str(znum)+'z', 'wb') as fout:
        pickle.dump(savedict, fout)
    
    # save to R
    pdzfinal = pd.DataFrame( np.append( y[...,None], zfinal, 1) )
    pdzfinal.to_csv('datasets//zfinal.csv', sep=',', encoding='utf-8', header=False, index=False)
    # to save to csv
    pdcombX = pd.DataFrame( np.append( y[...,None], combX, 1) )
    pdcombX.to_csv('datasets//combX.csv', sep=',', encoding='utf-8', header=False, index=False)
        
        
