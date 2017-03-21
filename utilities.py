# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:51:50 2017

@author: DeepLearning
"""

import os
import sys
#import cv2
#import cv
import numpy as np
from xml.dom import minidom

import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist

info_path = 'Z://Cristina//Section3//breast_MR_NME_pipeline' # os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path = [info_path] + sys.path
from query_localdatabase import *

def vis_square(fname, data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    data = data.mean(axis = -1)
    
    plt.imshow(data) 
    plt.savefig(fname)

def vis_cluster(dist, patch_dims, ntop, img):
    cluster = [ [] for i in xrange(dist.shape[1]) ]
    for i in xrange(dist.shape[0]):
        for j in xrange(dist.shape[1]):
            cluster[j].append((i, dist[i,j]))
    
    cluster.sort(key = lambda x: len(x), reverse = True)
    for i in cluster:
        print len(i)
        i.sort(key = lambda x: x[1], reverse=True)
    viz = np.zeros((patch_dims[0]*len(cluster), patch_dims[1]*ntop, img.shape[-1]))
    
    for i in xrange(len(cluster)):
        for j in xrange(min(ntop, len(cluster[i]))):
            viz[i*patch_dims[0]:(i+1)*patch_dims[0], j*patch_dims[1]:(j+1)*patch_dims[1], :] = img[cluster[i][j][0]]

    cv2.imwrite('viz_cluster.jpg', viz)
    
    
def vis_gradient(X, tmm, img):
    from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
    
    #with open('tmp.pkl') as fin:
    #    X, tmm, img = cPickle.load(fin)    
    #img = np.tile(img, 3)
    l = []
    q = tmm.transform(X)
    # pick a mu cluster equal to ind, based on the min number of counts, ind = 5
    # or pick a random index eg. ind=9
    ind = np.bincount(q.argmax(axis=1)).argmin()
    # select those q assignments to cluster 5
    l = [ i for i in xrange(X.shape[0]) if q[i].argmax() == ind ]
    # select corresponding X and images
    Xind = X[l,:]
    imgind = np.asarray(img)[l]

    # again find qs of Xs of cluster assignments to cluster 5
    q = tmm.transform(Xind)
    q = (q.T/q.sum(axis=1)).T
    # calculate target probabilities based on q
    p = (q**2)
    p = (p.T/p.sum(axis=1)).T
    # calculate begining of gradient
    grad = 2.0/(1.0+cdist(Xind, tmm.cluster_centers_, 'sqeuclidean'))*(p-q)*cdist(Xind, tmm.cluster_centers_, 'cityblock')

    fig, ax = plt.subplots()
    ax.scatter(q[:,ind], grad[:,ind], marker=u'+')

    n_disp = 10
    # sort the indices from large to small qs
    arg = np.argsort(q[:,ind])
    for i in xrange(n_disp):
        j = arg[int(Xind.shape[0]*(1.0-1.0*i/n_disp))-1]
        imgbox = OffsetImage(imgind[j], zoom=1)
        print q[j,ind]
        ab = AnnotationBbox(imgbox, (q[j,ind], grad[j,ind]),
                            xybox=(0.95-1.0*i/n_disp, 1.06 ),
                            xycoords='data',
                            boxcoords=("axes fraction", "axes fraction"),
                            pad=0.0,
                            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)
        
    plt.xlabel(r'$q_{ij}$', fontsize=24)
    plt.ylabel(r'$|\frac{\partial L}{\partial z_i}|$', fontsize=24)
    plt.draw()
    plt.show()
    

def vis_gradient_NME(combX, tmm, imgd, nxGdata, titleplot):
    from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import ConnectionPatch
    import scipy.spatial as spatial
    
    l = []
    q = tmm.transform(combX)
    # pick a mu cluster equal to ind, based on the min number of counts, ind = 5
    # or pick a random index eg. ind=9
    ind = np.bincount(q.argmax(axis=1)).argmax()
    # select those q assignments to cluster 5
    l = [ i for i in xrange(combX.shape[0]) if q[i].argmax() == ind ]
    # select corresponding X and images
    Xind = combX[l,:]
    imgind = np.asarray(imgd)[l]
    nxGind = nxGdata.iloc[l,:]

    # again find qs of Xs of cluster assignments to cluster 5
    q = tmm.transform(Xind)
    q = (q.T/q.sum(axis=1)).T
    # calculate target probabilities based on q
    p = (q**2)
    p = (p.T/p.sum(axis=1)).T
    # calculate begining of gradient
    grad = 2.0/(1.0+cdist(Xind, tmm.cluster_centers_, 'sqeuclidean'))*(p-q)*cdist(Xind, tmm.cluster_centers_, 'cityblock')

    ## plot
    fig = plt.figure(figsize=(20, 15))
    G = gridspec.GridSpec(4, 10)
    # for scatter
    ax0 = plt.subplot(G[1:4, 0:10])
    # fo exemplars 
    ax1 = plt.subplot(G[0,0]);     ax2 = plt.subplot(G[0,1])
    ax3 = plt.subplot(G[0,2]);     ax4 = plt.subplot(G[0,3])
    ax5 = plt.subplot(G[0,4]);     ax6 = plt.subplot(G[0,5])
    ax7 = plt.subplot(G[0,6]);     ax8 = plt.subplot(G[0,7])
    ax9 = plt.subplot(G[0,8]);     ax10 = plt.subplot(G[0,9])
    axes = [ax10,ax9,ax8,ax7,ax6,ax5,ax4,ax3,ax2,ax1]
        
    n_disp = 10
    # sort the indices from large to small qs
    arg = np.argsort(q[:,ind])
    ax0.scatter(q[:,ind], grad[:,ind], marker=u'+')
    for i in xrange(n_disp):
        j = arg[int(Xind.shape[0]*(1.0-1.0*i/n_disp))-1]
        ##
        ax = axes[i]
        ax.imshow(imgind[j], cmap=plt.cm.gray)
        ax.set_title('{}'.format(nxGind.iloc[j]['roiBIRADS']+nxGind.iloc[j]['classNME']))
        ax.get_xaxis().set_visible(False)                             
        ax.get_yaxis().set_visible(False)
        ##
        con = ConnectionPatch(xyA=(0,0), xyB=(q[j,ind], grad[j,ind]), 
                              coordsA='axes fraction', 
                              coordsB='data',
                              axesA=ax, axesB=ax0, 
                              arrowstyle="simple",connectionstyle='arc3')
        ax.add_artist(con)  
        
    ax0.set_xlabel(r'$q_{ij}$', fontsize=24)
    ax0.set_ylabel(r'$|\frac{\partial L}{\partial z_i}|$', fontsize=24)
    plt.draw()
    plt.show()
    fig.savefig( titleplot+'.pdf' )     

    
def vis_topscoring(X, tmm, img, pfinal, zfinal):
    from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
       
    # select those q assignments to cluster ind
    fig = plt.figure(figsize=(20, 15))
    n_disp = 10 

    for ind in xrange(n_disp):
        # select those q assignments to cluster ind
        l = [ i for i in xrange(X.shape[0]) if pfinal[i].argmax() == ind ]
        # select corresponding X and images
        Xind = X[l,:]
        imgind = np.asarray(img)[l]
        pfinalind = pfinal[l,:]
        
        # sort the indices from small to large p
        arg = np.argsort(pfinalind[:,ind])
        
        # plot top 10 scoreing
        for j in xrange(n_disp):
            k = arg[int(Xind.shape[0]-1-j)]
            # plot            
            row = ind*n_disp
            ax = fig.add_subplot(10,10,row+1+j)
            ax.imshow(imgind[k], cmap=plt.cm.gray_r)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_adjustable('box-forced')
            ax.set_xlabel("pij=%.3g"% pfinalind[k,ind])

    plt.draw()
    plt.show()
    # finally save it if not alreayd           
    plt.tight_layout()
    #fig.savefig( titleplot+'.pdf' )     
 
 
def vis_topscoring_NME(combX, imgd, num_centers, nxGdata, pfinal, zfinal, titleplot):
    from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
    # select those q assignments to cluster ind
    fig = plt.figure(figsize=(20, 15))
    n_disp = 5

    for ind in xrange(num_centers):
        # select those q assignments to cluster ind
        l = [ i for i in xrange(combX.shape[0]) if pfinal[i].argmax() == ind ]
        # select corresponding X and images
        Xind = combX[l,:]
        imgind = np.asarray(imgd)[l]
        pfinalind = pfinal[l,:]
        nxGind = nxGdata.iloc[l,:]

        # sort the indices from small to large p
        arg = np.argsort(pfinalind[:,ind])
        
        num_incluster = len(l)
        if(num_incluster < n_disp):
            n_disp = num_incluster
            
        # plot top 10 scoreing
        for j in xrange(n_disp):
            k = arg[int(Xind.shape[0]-1-j)]
            # plot            
            row = ind*n_disp
            ax = fig.add_subplot(num_centers,n_disp,row+1+j)
            ax.imshow(imgind[k], cmap=plt.cm.gray_r)
            ax.set_adjustable('box-forced')
            ax.set_title('{}_{}'.format(nxGind.iloc[k]['roi_id'],nxGind.iloc[k]['roiBIRADS']+nxGind.iloc[k]['classNME']+str(nxGind.iloc[k]['nme_dist'])))
            ax.get_xaxis().set_visible(False)                             
            ax.get_yaxis().set_visible(False)

    plt.draw()
    plt.show()
    # finally save it if not alreayd           
    plt.tight_layout()
    fig.savefig( titleplot+'.pdf' )     
    
    
    
class TMM(object):
    def __init__(self, n_components=1, alpha=1): 
        self.n_components = n_components
        self.tol = 1e-5
        self.alpha = float(alpha)
        
    def fit(self, X):
        from sklearn.cluster import KMeans
        kmeans = KMeans(self.n_components, n_init=20)
        kmeans.fit(X)
        self.cluster_centers_ = kmeans.cluster_centers_
        self.covars_ = np.ones(self.cluster_centers_.shape)
    
    def transform(self, X):
        p = 1.0
        dist = cdist(X, self.cluster_centers_)
        r = 1.0/(1.0+dist**2/self.alpha)**((self.alpha+p)/2.0)
        r = (r.T/r.sum(axis=1)).T
        return r
    
    def predict(self, X):
        return self.transform(X).argmax(axis=1)
    
    
def plot_embedding(X, y, ax, title=None, legend=True, plotcolor=True):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import offsetbox
    from matplotlib.offsetbox import TextArea, AnnotationBbox
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # process labels 
    classes = [str(c) for c in np.unique(y)]
    colors=plt.cm.viridis(np.linspace(0,1,len(classes))) # plt.cm.gist_rainbow
    c_patchs = []
    for k in range(len(classes)):
            c_patchs.append(mpatches.Patch(color=colors[k], label=classes[k]))
        
    if(legend):
        plt.legend(handles=c_patchs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':10})
    

    for i in range(X.shape[0]):
        for k in range(len(classes)):
            if str(y[i])==classes[k]: 
                colori = colors[k] 
        
        if(plotcolor):                
            plt.text(X[i, 0], X[i, 1], str(y[i])[0], color=colori,
                     fontdict={'weight': 'bold', 'size': 8})
        else:
            greycolor = plt.cm.Accent(1)    
            plt.text(X[i, 0], X[i, 1], '.', color=greycolor,
                         fontdict={'weight': 'bold', 'size': 10})

    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    if title is not None:
        plt.title(title)
        
        
def visualize_graph_ndMRIdata(roi_id, typenxg, colorlegend):
    '''
    # Construct img dictionary calling visualize_graph_ndMRIdata(roi_id) per roi_id
    from utilities import visualize_graph_ndMRIdata
    for roi_id in range(1,len(nxGdata)+1):
        visualize_graph_ndMRIdata(roi_id, typenxg='MST', colorlegend=False)
    '''
    import glob
    import six.moves.cPickle as pickle
    import gzip
    import SimpleITK as sitk
    import networkx as nx
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ################################### 
    # to visualize graphs and MRI data
    ###################################
    graphs_path = 'Z:\\Cristina\\Section3\\breast_MR_NME_pipeline\\processed_NMEs'
    # to read 4th post-C registered MRI to pre-contrast
    processed_path = r'Z:\Cristina\Section3\breast_MR_NME_pipeline\processed_data'

    ###### 1) Querying Research database for clinical, pathology, radiology data
    #roi_id = nxGdata[nxGdata['roi_id']==1].iloc[0]['roi_id']
    
    # get lesion info from database  
    localdata = Querylocal()
    dflesion  = localdata.queryby_roid(roi_id)
    
    nmlesion_record = pd.Series(dflesion.Nonmass_record.__dict__)       
    lesion_record = pd.Series(dflesion.Lesion_record.__dict__)
    roi_record = pd.Series(dflesion.ROI_record.__dict__)
    #lesion_id = lesion_record['lesion_id']
    StudyID = lesion_record['cad_pt_no_txt']
    AccessionN = lesion_record['exam_a_number_txt']
    DynSeries_id = nmlesion_record['DynSeries_id'] 
    roiLabel = roi_record['roi_label']
    zslice = int(roi_record['zslize'])
    p1 = roi_record['patch_diag1']
    patch_diag1 = p1[p1.find("(")+1:p1.find(")")].split(',')
    patch_diag1 = [float(p) for p in patch_diag1]
    p2 = roi_record['patch_diag2']
    patch_diag2 = p2[p2.find("(")+1:p2.find(")")].split(',')
    patch_diag2 = [float(p) for p in patch_diag2]    
    ext_x = [int(ex) for ex in [np.min([patch_diag1[0],patch_diag2[0]])-20,np.max([patch_diag1[0],patch_diag2[0]])+20] ] 
    ext_y = [int(ey) for ey in [np.min([patch_diag1[1],patch_diag2[1]])-20,np.max([patch_diag1[1],patch_diag2[1]])+20] ] 
    
    ###### 2) Accesing mc images, prob maps, gt_lesions and breast masks
    precontrast_id = int(DynSeries_id) 
    DynSeries_nums = [str(n) for n in range(precontrast_id,precontrast_id+5)]

    print "Reading MRI 4th volume..."
    try:
        #the output mha:lesionid_patientid_access#_series#@acqusionTime.mha
        DynSeries_filename = '{}_{}_{}'.format(StudyID.zfill(4),AccessionN,DynSeries_nums[4] )
        glob_result = glob.glob(os.path.join(processed_path,DynSeries_filename+'*')) 
        if glob_result != []:
            filename = glob_result[0]
        # read Volumnes
        mriVolDICOM = sitk.ReadImage(filename)
        mri4th = sitk.GetArrayFromImage(sitk.Cast(mriVolDICOM,sitk.sitkFloat32)) 
     
    except:
        logger.info('   failed: locating dynSeries w motion_correction!')
        roi_id = roi_id+1
        return -1
    
    ###### 3) load DEL and MST graph object into memory
    if(typenxg=='DEL'):
        try:
            with gzip.open( os.path.join(graphs_path,'{}_{}_{}_{}_FacesTriang_lesion_nxgraph.pklz'.format(str(roi_id),StudyID.zfill(4),AccessionN,roiLabel)), 'rb') as f:
                nxGraph = pickle.load(f)
        except:
            filegraph = glob.glob( os.path.join(graphs_path,'{}_{}_{}_*_FacesTriang_lesion_*'.format(str(roi_id),StudyID.zfill(4),AccessionN) ))
            with gzip.open( filegraph[0], 'rb') as f:
                nxGraph = pickle.load(f)
        nxGraph_name = 'DEL_'+str(roi_id)
        
    if(typenxg=='MST'):
        try:
            with gzip.open( os.path.join(graphs_path,'{}_{}_{}_{}_MST_lesion_nxgraph.pklz'.format(str(roi_id),StudyID.zfill(4),AccessionN,roiLabel)), 'rb') as f:
                nxGraph = pickle.load(f)
        except:
            filegraph = glob.glob( os.path.join(graphs_path,'{}_{}_{}_*_MST_*'.format(str(roi_id),StudyID.zfill(4),AccessionN) ))
            with gzip.open( filegraph[0], 'rb') as f:
                nxGraph = pickle.load(f)
        nxGraph_name = 'MST_'+str(roi_id)
                   
    ###### 4) plot MRI + graph
    # The triangles in parameter space determine which x, y, z points are connected by an edge
    fig, ax = plt.subplots(dpi=200)   
    # show MRI slice
    ax.imshow(mri4th[zslice,:,:], cmap=plt.cm.gray)
    ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
     # draw
    MST_nodeweights = [d['weight'] for (u,v,d) in nxGraph.edges(data=True)]
    MST_pos = np.asarray([p['pos'] for (u,p) in nxGraph.nodes(data=True)])

    nxg = nx.draw_networkx_edges(nxGraph, MST_pos, ax=ax, edge_color=MST_nodeweights, edge_cmap=plt.cm.inferno, 
                             edge_vmin=-0.01,edge_vmax=2.5, width=1.5)
    ax.set_adjustable('box-forced')
    ax.get_xaxis().set_visible(False)                             
    ax.get_yaxis().set_visible(False)
    
    # add color legend
    if(colorlegend):
        v = np.linspace(-0.01, 2.5, 10, endpoint=True)     
        divider = make_axes_locatable(ax)
        caxEdges = divider.append_axes("right", size="20%", pad=0.05)
        plt.colorbar(nxg, cax=caxEdges, ticks=v) 

    # save
    fig.savefig('figs//'+nxGraph_name+'.png', bbox_inches='tight')    
    plt.close()

    return


def make_graph_ndMRIdata(roi_id, typenxg):
    import matplotlib.pyplot as plt
    import numpy as np  
    from matplotlib._png import read_png
    
    ###### 1) read DEL and MST png
    if(typenxg=='DEL'):
        nxGraph_name = 'DEL_'+str(roi_id)
        
    if(typenxg=='MST'):
       nxGraph_name = 'MST_'+str(roi_id)       
       
    #img = plt.imread('figs//'+nxGraph_name+'.png')
    img = read_png('figs//'+nxGraph_name+'.png')
    
    return img
    

def plot_pngs_showNN(tsne_id, Z_tsne, y_tsne, lesion_id, nxG_name, title=None):
    '''Scale and visualize the embedding vectors
    version _showG requires additional inputs like lesion_id and corresponding mriVol    
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import ConnectionPatch
    import scipy.spatial as spatial

    ########################################   
    fig_path = r'Z:\Cristina\Section3\exploreDR_NME_classification\mxnet\NME_DEC\figs'

    x_min, x_max = np.min(Z_tsne, 0), np.max(Z_tsne, 0)
    Z_tsne = (Z_tsne - x_min) / (x_max - x_min)

    figTSNE = plt.figure(figsize=(32, 24))
    G = gridspec.GridSpec(4, 4)
    # for tsne
    ax1 = plt.subplot(G[0:3, 0:3])
    # fo lesion id graph
    ax2 = plt.subplot(G[0,3])
    # plot for neighbors
    ax3 = plt.subplot(G[1,3])
    ax4 = plt.subplot(G[2,3])
    ax5 = plt.subplot(G[3,3])
    ax6 = plt.subplot(G[3,2])
    ax7 = plt.subplot(G[3,1])
    ax8 = plt.subplot(G[3,0])
    axes = [ax3,ax4,ax5,ax6,ax7,ax8]
    
    # turn axes off
    ax2.get_xaxis().set_visible(False)                             
    ax2.get_yaxis().set_visible(False)   
    for ax in axes:
        ax.get_xaxis().set_visible(False)                             
        ax.get_yaxis().set_visible(False)          

    # process labels 
    classes = [str(c) for c in np.unique(y_tsne)]
    colors=plt.cm.rainbow(np.linspace(0,1,len(classes)))
    c_patchs = []
    for k in range(len(classes)):
         c_patchs.append(mpatches.Patch(color=colors[k], label=classes[k]))
    ax1.legend(handles=c_patchs, loc='center right', bbox_to_anchor=(-0.05,0.5), ncol=1, prop={'size':12})
    ax1.grid(True)
    ax1.set_xlim(-0.1,1.1)
    ax1.set_ylim(-0.1,1.1)    
    
    ## plot tsne
    for i in range(Z_tsne.shape[0]):
        for k in range(len(classes)):
            if str(y_tsne[i])==classes[k]: 
                colori = colors[k] 
        ax1.text(Z_tsne[i, 0], Z_tsne[i, 1], str(y_tsne[i]), color=colori,
                 fontdict={'weight': 'bold', 'size': 8})     
                 
    #############################
    ###### 2) load MST and display (png) 
    #############################
    # us e ConnectorPatch is useful when you want to connect points in different axes
    con1 = ConnectionPatch(xyA=(0,1), xyB=Z_tsne[tsne_id-1], coordsA='axes fraction', coordsB='data',
            axesA=ax2, axesB=ax1, arrowstyle="simple",connectionstyle='arc3')
    ax2.add_artist(con1)                   

    img_ROI = read_png( os.path.join(fig_path,'MST_'+str(tsne_id)+'.png') )
    ax2.imshow(img_ROI, cmap=plt.cm.gray)
    ax2.set_adjustable('box-forced')   
    ax2.set_title(nxG_name)

         
    #############################
    ###### 3) Find closest neighborhs and plot
    #############################
    pdNN = pd.DataFrame({})  
    
    Z_embedding_tree = spatial.cKDTree(Z_tsne, compact_nodes=True)
    # This finds the index of all points within distance 0.1 of embedded point X_tsne[lesion_id]
    closestd = 0.01
    NN_embedding_indx_list = Z_embedding_tree.query_ball_point(Z_tsne[tsne_id-1], closestd)
    
    while(len(NN_embedding_indx_list)<=5):
        closestd+=0.005
        NN_embedding_indx_list = Z_embedding_tree.query_ball_point(Z_tsne[tsne_id-1], closestd)

    NN_embedding_indx = [knn for knn in NN_embedding_indx_list if knn != tsne_id-1]
    k_nn = min(6,len(NN_embedding_indx))
    
    # plot knn embedded poitns
    for k in range(k_nn):
        k_nn_roid_indx = NN_embedding_indx[k] # finds indices from 0-614, but roi_id => 1-615
        
        # us e ConnectorPatch is useful when you want to connect points in different axes
        conknn = ConnectionPatch(xyA=(0,1), xyB=Z_tsne[k_nn_roid_indx], coordsA='axes fraction', coordsB='data',
                axesA=axes[k], axesB=ax1, arrowstyle="simple",connectionstyle='arc3')
        axes[k].add_artist(conknn) 
            
        ###### read MST from roi_id
        k_nn_roid = k_nn_roid_indx+1  # finds indices from 0-614, but roi_id => 1-615
        knn_img_ROI = read_png( os.path.join(fig_path,'MST_'+str(k_nn_roid)+'.png') )
        axes[k].imshow(knn_img_ROI, cmap=plt.cm.gray)
        axes[k].set_adjustable('box-forced')
        
        ###### find database info for roi_id
        localdata = Querylocal()
        dflesionk_nn  = localdata.queryby_roid(k_nn_roid)
        
        lesion_record = pd.Series(dflesionk_nn.Lesion_record.__dict__)
        roi_record = pd.Series(dflesionk_nn.ROI_record.__dict__)
        
        #lesion_id = lesion_record['lesion_id']
        knn_lesion_id = lesion_record['lesion_id']
        StudyID = lesion_record['cad_pt_no_txt']
        AccessionN = lesion_record['exam_a_number_txt']    
        roiLabel = roi_record['roi_label']
        roi_diagnosis = roi_record['roi_diagnosis']
        roi_BIRADS = lesion_record['BIRADS']
        print "Indication by y_tsne: %s, by database: %s = %s" % (y_tsne[k_nn_roid_indx],roi_BIRADS+roiLabel,roi_diagnosis)
 
        #############################
        ###### 3) Examine and plot TSNE with KNN neighbor graphs in a radius of tnse embedding = 0.1
        #############################         
        nxG_name = '{}_{}_roi{}_lesion{}'.format(y_tsne[k_nn_roid_indx],roi_diagnosis,k_nn_roid,str(knn_lesion_id))
        axes[k].set_title(nxG_name)  
            
        ## append to dataframe of neighbors pdNN
        #############################
        # to build dataframe    
        rows = []; index = []     
        rows.append({'k_nn_roid_indx': k_nn_roid_indx,
                     'k_nn_roid': k_nn_roid,
                     'knn_lesion_id': knn_lesion_id,
                     'fStudyID': StudyID,
                     'AccessionN':AccessionN,                         
                     'class': roi_BIRADS+roiLabel, 
                     'type': roi_diagnosis})         
        index.append(str(k_nn_roid))
            
        # append counts to master lists
        pdNN = pdNN.append( pd.DataFrame(rows, index=index) )
        #############################
               
    if title is not None:
        plt.title(title)
    
    plt.tight_layout()
    return figTSNE, pdNN
        
    
def visualize_Zlatent_NN_fortsne_id(Z_tsne, y_tsne, tsne_id, saveFigs=False):
    # Get Root folder ( the directory of the script being run)
    import sys
    import glob
    import six.moves.cPickle as pickle
    import gzip
    import SimpleITK as sitk
    import networkx as nx
    from matplotlib._png import read_png
    import matplotlib.pyplot as plt    

    #############################
    ###### 1) Querying Research database for clinical, pathology, radiology data
    ############################# 
    localdata = Querylocal()
    dflesion  = localdata.queryby_roid(tsne_id)
    
    lesion_record = pd.Series(dflesion.Lesion_record.__dict__)
    nmlesion_record = pd.Series(dflesion.Nonmass_record.__dict__)       
    roi_record = pd.Series(dflesion.ROI_record.__dict__)
    
    #lesion_id = lesion_record['lesion_id']
    lesion_id = lesion_record['lesion_id']
    StudyID = lesion_record['cad_pt_no_txt']
    AccessionN = lesion_record['exam_a_number_txt']
    DynSeries_id = nmlesion_record['DynSeries_id'] 
    
    roiLabel = roi_record['roi_label']
    roi_diagnosis = roi_record['roi_diagnosis']
    roi_BIRADS = lesion_record['BIRADS']
    print "Indication by y_tsne: %s, by database: %s = %s" % (y_tsne[tsne_id-1],roi_BIRADS+roiLabel,roi_diagnosis)
    print "Querying tsne_id %i, lesion_id=%s, fStudyID=%s, AccessionN=%s, sideB=%s" % (tsne_id, lesion_id, StudyID, AccessionN, DynSeries_id)
    print "has the following upto 5 nearest-neighbors:"

    #############################
    ###### 3) Examine and plot TSNE with KNN neighbor graphs in a radius of tnse embedding = 0.1
    #############################         
    nxG_name = '{}_{}_roi{}_lesion{}'.format(y_tsne[tsne_id-1],roi_diagnosis,tsne_id,str(lesion_id))
    figTSNE, pdNN = plot_pngs_showNN(tsne_id, Z_tsne, y_tsne, lesion_id, nxG_name, title=None)   
    
    #show and save
    if(saveFigs):
        figTSNE.savefig( os.path.join(nxGfeatures_path,'lesion_id_{}_TSNE_nxGwSER_{}{}_{}.pdf'.format(str(lesion_id),cond,BenignNMaligNAnt,Diagnosis)), bbox_inches='tight') 
        plt.close()
   
    return pdNN    
        