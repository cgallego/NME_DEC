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
        sep = int(X.shape[0]*0.75)
        X_train = X[:sep]
        X_val = X[sep:]
        batch_size = 160 #32*5 = update_interval*5
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

    def tsne_wtest(self, X_test, y_wtest, zfinal):
        ## self=dec_model
        ## embedded point zi and
        N_test = X_test.shape[0] #X_test.transpose().shape
        batch_size = 1 # 41  #256
        test_iter = mx.io.NDArrayIter({'data': X_test}, batch_size=batch_size, shuffle=False,
                                      last_batch_handle='pad')
        args = {k: mx.nd.array(v.asnumpy(), ctx=self.xpu) for k, v in self.args.items()}
        z_test = model.extract_feature(self.feature, args, None, test_iter, N_test, self.xpu).values()[0]
        
        z_wtest = np.vstack([zfinal, z_test])
        
        # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
        tsne = TSNE(n_components=2, perplexity=15, learning_rate=375,
             init='pca', random_state=0, verbose=2, method='exact')
        Z_tsne = tsne.fit_transform(z_wtest)
        
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plot_embedding(Z_tsne, y_wtest, ax, title="tsne with test y class(%d)" % (y_wtest[-1]), legend=True, plotcolor=True)
        #fig.savefig('results//tsne_wtest_NME_k'+str(num_centers)+'.pdf', bbox_inches='tight')    
        #plt.close()            
        
    def cluster(self, X, y=None, update_interval=None):
        # 
        N = X.shape[0]
        if not update_interval:
            update_interval = int(self.batch_size/5.0)
            
        # selecting batch size
        # [42*t for t in range(42)]  will produce 16 train epochs
        # [0, 42, 84, 126, 168, 210, 252, 294, 336, 378, 420, 462, 504, 546, 588, 630]
        batch_size = self.batch_size #615/3 42  #256
        test_iter = mx.io.NDArrayIter({'data': X}, 
                                      batch_size=batch_size, 
                                      shuffle=False,
                                      last_batch_handle='pad')
        args = {k: mx.nd.array(v.asnumpy(), ctx=self.xpu) for k, v in self.args.items()}
        ## embedded point zi and
        z = model.extract_feature(self.feature, args, None, test_iter, N, self.xpu).values()[0]
        
        # For visualization we use t-SNE (van der Maaten & Hinton, 2008) applied to the embedded points zi. It
        pp = 15
        tsne = TSNE(n_components=2, perplexity=pp, learning_rate=375,
             init='pca', random_state=0, verbose=2, method='exact')
        Z_tsne = tsne.fit_transform(z)
        
        # plot initial z        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plot_embedding(Z_tsne, y, ax, title="tsne with perplexity %d" % pp, legend=True, plotcolor=True)
        fig.savefig('results//tsne_init_k'+str(self.num_centers)+'_z'+str(self.znum)+'.pdf', bbox_inches='tight')    
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
                    plot_embedding(Z_tsne, y, ax, title="Epoch %d z_tsne iter (%d)" % (self.ploti,i), legend=False, plotcolor=True)
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
                   'z': self.z}
            
        return outdict



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
    
    # Build/Read AE
    y = YnxG[0] #+YnxG[1]
    num_centers = len(np.unique(y)) # present but not needed during AE training
    ae_znum = [10,15,20,30,40,50]
    for znum in ae_znum:
        print "Building autoencoder of latent size znum = ",znum
        dec_model = DECModel(mx.cpu(), combX, num_centers, 1.0, znum, 'model/NME_'+str(znum)+'k')
    
    # preparation for fig 5.
    from utilities import make_graph_ndMRIdata
    imgd = []
    for roi_id in range(1,len(nxGdata)+1):   
        imgd.append( make_graph_ndMRIdata(roi_id, typenxg='MST'))
    
    # Plot fig 5: Gradient vs. soft assigments before KL divergence
    tmm = TMM(n_components=num_centers, alpha=1.0)
    tmm.fit(combX)
    vis_gradient_NME(combX, tmm, imgd, nxGdata, titleplot='results//vis_gradient_BIRADS_k'+str(num_centers)+'_z'+str(znum))
    
    ## use y_dec to  minimizing KL divergence for clustering
    y = YnxG[0]#+YnxG[1]
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
    ## Quanlitative evaluation        
    ## cluster DEC for BIRADS score prediction
    ########################################################
    ## visualize the progression of the embedded representation of a random subset data 
    # during training. For visualization we use t-SNE (van der Maaten & Hinton, 2008) 
    # applied to the embedded points z
    num_centers = len(np.unique(y)) # present but not needed during AE training
    ae_znum = [10,15,20,30,40,50]
    clusteringAc_znum = []
    for znum in ae_znum:
        print "Runing DEC with autoencoder of latent size znum = ",znum
        dec_model = DECModel(mx.cpu(), combX, num_centers, 1.0, znum, 'model/NME_'+str(znum)+'k')
        outdict = dec_model.cluster(combX, y_dec, update_interval=32)
        # save
        with open('model/NME_clusters_'+str(znum)+'k', 'wb') as fout:
            pickle.dump(outdict, fout)
        # to load  
#        with open('model/NME_clusters_'+str(znum)+'k','rb') as fin:
#            outdict = pickle.load(fin)
            
        fig = plt.gcf()
        fig.savefig('results//tsne_progress_k'+str(num_centers)+'_z'+str(znum)+'.pdf', bbox_inches='tight')    
        plt.close()
    
        pfinal = outdict['p']
        zfinal = outdict['z']
        clusteringAc_znum = 
        # to get final cluster memberships
        ypredfinal = pfinal.argmax(axis=1)
        vis_topscoring_NME(combX, imgd, num_centers, nxGdata, pfinal, zfinal, titleplot='results//vis_topscoring_k'+str(num_centers)+'_z'+str(znum))
        
        # plot final z  
        tsne = TSNE(n_components=2, perplexity=15, learning_rate=375,
                 init='pca', random_state=0, verbose=2, method='exact')
        Z_tsne = tsne.fit_transform(zfinal)
              
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plot_embedding(Z_tsne, y, ax, title="tsne embedding space after DEC", legend=True, plotcolor=True)
        fig.savefig('results//tsne_final_k'+str(num_centers)+'_z'+str(znum)+'.pdf', bbox_inches='tight')    
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
        tsne_id=36   ## e.g roi_id = 36 DUCTAL CARCINOMA IN-SITU
        y_tsne = y #YnxG[0]+YnxG[1]
        tsne_id_class, tsne_id_type, tsne_id_Diagnosis, pdNN = visualize_Zlatent_NN_fortsne_id(Z_tsne, y_tsne, tsne_id, saveFigs=True)



    ########################################################    
    ## A test case, obtain z representation and map
    # e.g
    X_test = combX[0,:]
    X_test = np.reshape(X_test, (1, X_test.shape[0]))
    y_test = y_dec[0]
    y_wtest = list(y_dec)
    y_wtest.append(y_test)
    y_wtest = np.asarray(y_wtest)
    #X_wtest = np.vstack([combX, X_atest])
    
    dec_model.tsne_wtest(X_test, y_test, y_wtest, zfinal)
    