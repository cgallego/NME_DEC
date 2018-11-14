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

#info_path = 'Z://Cristina//Section3//breast_MR_NME_pipeline' # os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
#sys.path = [info_path] + sys.path
#from query_localdatabase import *

class TMM(object):
    from scipy.spatial.distance import cdist
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
    
    
def vis_absgradient_final(pfinal, zfinal, dec_mu, sep):
  
    qfinal_train = pfinal[:sep,:]
    qfinal_val = pfinal[sep:,:]
    # calculate target probabilities based on q
    pfinal_train = (qfinal_train**2)
    pfinal_train = (pfinal_train.T/pfinal_train.sum(axis=1)).T
    pfinal_val = (qfinal_val**2)
    pfinal_val = (pfinal_val.T/pfinal_val.sum(axis=1)).T
    # calculate begining of gradient
    gradfinal_train = 2.0/(1.0+cdist(zfinal[:sep,:], dec_mu, 'sqeuclidean'))*(pfinal_train-qfinal_train)*cdist(zfinal[:sep,:], dec_mu, 'cityblock')
    gradfinal_val = 2.0/(1.0+cdist(zfinal[sep:,:], dec_mu, 'sqeuclidean'))*(pfinal_val-qfinal_val)*cdist(zfinal[sep:,:], dec_mu, 'cityblock')

    G = np.mean(gradfinal_train.mean(axis=0))/ np.mean(gradfinal_val.mean(axis=0))
    return(G)
    
  
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
    disp_num_centers = min(5,num_centers)

    # find cluster assigments and sort to display the 5 bigest clusters    
    n_clusterind = []
    for ind in xrange(num_centers):
        # select those q assignments to cluster ind
        l = [ i for i in xrange(combX.shape[0]) if pfinal[i].argmax() == ind ]
        n_clusterind.append( len(l) )
        
    # select clusters to display
    sel_clusters = np.argsort(n_clusterind)[::-1][:disp_num_centers]
    for kind, ind in enumerate(sel_clusters):
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
            row = kind*n_disp
            ax = fig.add_subplot(disp_num_centers,n_disp,row+1+j)
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
    greyc_U = np.array([0.5,0.5,0.5,1])
    for k in range(len(classes)):
        if(classes[k]=="U" or classes[k]=="N/A"):
            c_patchs.append(mpatches.Patch(color=greyc_U, label=classes[k]))
        else:
            c_patchs.append(mpatches.Patch(color=colors[k], label=classes[k]))
        
    if(legend):
        plt.legend(handles=c_patchs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':10})
    
    for i in range(X.shape[0]):
        for k in range(len(classes)):
            if str(y[i])==classes[k]: 
                if(classes[k]=="U" or classes[k]=="N/A"):
                    colori = greyc_U
                else:
                    colori = colors[k] 
        
        if(plotcolor):          
            plotlabel = [s[0] for s in y[i].split('_')]
            plt.text(X[i, 0], X[i, 1], '.', color=colori, # optimonal label = ''.join(plotlabel)
                     fontdict={'weight': 'bold', 'size': 24})
        else:
            greycolor = plt.cm.Accent(1)    
            plt.text(X[i, 0], X[i, 1], '.', color=greycolor,
                         fontdict={'weight': 'bold', 'size': 24})

    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    if title is not None:
        plt.title(title)

def plot_embedding_unsuper_NMEdist_intenh(Z_tsne, named_y, ax, title=None, legend=True):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import offsetbox
    from matplotlib.offsetbox import TextArea, AnnotationBbox
    
    x_min, x_max = np.min(Z_tsne, 0), np.max(Z_tsne, 0)
    Z_tsne = (Z_tsne - x_min) / (x_max - x_min)

    # process labels 
    classes = [str(c) for c in np.unique(named_y)]
    try:
        classes.remove('K_nan_nan')
    except:
        pass
    colors=plt.cm.viridis(np.linspace(0,1,len(classes))) # plt.cm.gist_rainbow
    c_patchs = []
    greyc_U = np.array([0.5,0.5,0.5,0.5])
    for k in range(len(classes)):
            c_patchs.append(mpatches.Patch(color=colors[k], label=classes[k]))

    for i in range(Z_tsne.shape[0]):
        for k in range(len(classes)):
            if str(named_y[i])==classes[k]: 
                colori = colors[k]      
        if(i<202):
            plt.text(Z_tsne[i, 0], Z_tsne[i, 1], str(named_y[i])[0], color=colori,
                     fontdict={'weight': 'bold', 'size': 8})
        else: 
            if(str(named_y[i])=='K_N/A_N/A'):
                #print('{}..{}'.format(i,str(named_y[i])))
                plt.text(Z_tsne[i, 0], Z_tsne[i, 1], '.', color=greyc_U,
                     fontdict={'weight': 'bold', 'size': 24}) 
                     
            elif(str(named_y[i])[0]!='K'):
                plt.text(Z_tsne[i, 0], Z_tsne[i, 1], str(named_y[i])[0], color=colori,
                     fontdict={'weight': 'bold', 'size': 8})
            else: 
                #print('{}..{}'.format(i,str(named_y[i])))
                plt.text(Z_tsne[i, 0], Z_tsne[i, 1], '.', color=colori,
                         fontdict={'weight': 'bold', 'size': 24})
            
    if(legend):
        c_patchs.append(mpatches.Patch(color=greyc_U, label='unknown'))
        plt.legend(handles=c_patchs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':8})
    

    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    if title is not None:
        plt.title(title)
        
        
def plot_embedding_unsuper(Z_tsne, named_y, ax, title=None, legend=True):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import offsetbox
    from matplotlib.offsetbox import TextArea, AnnotationBbox
    
    x_min, x_max = np.min(Z_tsne, 0), np.max(Z_tsne, 0)
    Z_tsne = (Z_tsne - x_min) / (x_max - x_min)

    # process labels 
    classes = [str(c) for c in np.unique(named_y)]
    #classes.remove('K_N/A')
    colors=plt.cm.viridis(np.linspace(0,1,len(classes))) # plt.cm.gist_rainbow
    c_patchs = []
    greyc_U = np.array([0.5,0.5,0.5,0.5])
    for k in range(len(classes)):
            c_patchs.append(mpatches.Patch(color=colors[k], label=classes[k]))

    for i in range(Z_tsne.shape[0]):
        for k in range(len(classes)):
            if str(named_y[i])==classes[k]: 
                colori = colors[k]      
        if(i<202):
            plt.text(Z_tsne[i, 0], Z_tsne[i, 1], str(named_y[i])[0], color=colori,
                     fontdict={'weight': 'bold', 'size': 8})
        else: 
            if(str(named_y[i])=='K_N/A'):
                #print('{}..{}'.format(i,str(named_y[i])))
                plt.text(Z_tsne[i, 0], Z_tsne[i, 1], '.', color=greyc_U,
                     fontdict={'weight': 'bold', 'size': 24}) 
                     
            elif(str(named_y[i])[0]!='K'):
                plt.text(Z_tsne[i, 0], Z_tsne[i, 1], str(named_y[i])[0], color=colori,
                     fontdict={'weight': 'bold', 'size': 8})
            else: 
                #print('{}..{}'.format(i,str(named_y[i])))
                plt.text(Z_tsne[i, 0], Z_tsne[i, 1], '.', color=colori,
                         fontdict={'weight': 'bold', 'size': 24})
            
    if(legend):
        c_patchs.append(mpatches.Patch(color=greyc_U, label='unknown'))
        plt.legend(handles=c_patchs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':10})
    

    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    if title is not None:
        plt.title(title)
 
def plot_embedding_unsuper_wFU(Z_tsne, y_tsne, ax, title=None, legend=True, withClustersImg=True):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import offsetbox
    from matplotlib.offsetbox import TextArea, AnnotationBbox
    
    x_min, x_max = np.min(Z_tsne, 0), np.max(Z_tsne, 0)
    Z_tsne = (Z_tsne - x_min) / (x_max - x_min)

    # process labels 
    classes = [str(c) for c in np.unique(y_tsne)]
    #classes.remove('K_N/A')
    colors=plt.cm.viridis(np.linspace(0,1,len(classes))) # plt.cm.gist_rainbow
    c_patchs = []
    greyc_U = np.array([0.5,0.5,0.5,0.5])
    for k in range(len(classes)):
        if(str(classes[k])!='FU'):
            c_patchs.append(mpatches.Patch(color=colors[k], label=classes[k]))
        else:
            c_patchs.append(mpatches.Patch(color=greyc_U, label='unknown'))

    for i in range(Z_tsne.shape[0]):
        for k in range(len(classes)):
            if str(y_tsne[i])==classes[k]: 
                colori = colors[k]     
                
        if(str(y_tsne[i])!='FU'):
            plt.text(Z_tsne[i, 0], Z_tsne[i, 1], str(y_tsne[i]), color=colori,
                     fontdict={'weight': 'bold', 'size': 10})
        else:        
            #print('{}..{}'.format(i,str(named_y[i])))
            plt.text(Z_tsne[i, 0], Z_tsne[i, 1], '.', color=greyc_U,
                 fontdict={'weight': 'bold', 'size': 24}) 
                     
            
    if(legend):
        plt.legend(handles=c_patchs, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':14})
    
    if(withClustersImg):
        # plot closets image to cluster centroid: one per class
        from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
        tmm = TMM(n_components=5, alpha=1.0)
        tmm.fit(Z_tsne)
        
        l_clusters = []
        Z_ind_clusters = []
        q = tmm.transform(Z_tsne)
        num_clusters=5
        for ind in range(num_clusters):
            # select each cluster
            l = [ i for i in xrange(Z_tsne.shape[0]) if q[i].argmax() == ind ]
            Z_ind = Z_tsne[l,:]
            l_clusters.append(l)
            Z_ind_clusters.append(Z_ind)
            
            # plot            
            ind_centroids = Z_ind.mean(axis=0)
            ax.annotate('cluster_'+str(ind), xy=(ind_centroids[0], ind_centroids[1]),
                xytext=(0.95-1.0*ind/num_clusters, 0.90),
                xycoords='data',
                textcoords="data",
                arrowprops=dict(arrowstyle="->"))

    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    if title is not None:
        plt.title(title) 
    
    return l_clusters, Z_ind_clusters

        
def visualize_graph_ndMRIdata(typenxg, colorlegend):
    '''
    # Construct img dictionary calling visualize_graph_ndMRIdata(roi_id) per roi_id
    from utilities import visualize_graph_ndMRIdata
    # to run    
    visualize_graph_ndMRIdata(typenxg='DEL', colorlegend=True)
    visualize_graph_ndMRIdata(typenxg='MST', colorlegend=True)
    '''
    import glob, sys, os
    import six.moves.cPickle as pickle
    import gzip
    import SimpleITK as sitk
    import networkx as nx
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sqlalchemy.orm import sessionmaker, joinedload_all
    from sqlalchemy import create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    info_path = 'Z://Cristina//Section3//NME_DEC//imgFeatures' 
    sys.path = [info_path] + sys.path
    import localdatabase
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt 

    ################################### 
    # to visualize graphs and MRI data
    ###################################
    graphs_path = 'Z:\\Cristina\\Section3\\NME_DEC\\imgFeatures\\processed_NMEs'
    
    lesion_id = 1
    while ( lesion_id <= 1232 ) :
        ###### 1) Querying Research database for clinical, pathology, radiology data         
        print "Executing SQL local connection database..."
        # configure Session class with desired options
        Session = sessionmaker()
        queryengine = create_engine('sqlite:///Z:\\Cristina\\Section3\\NME_DEC\\imgFeatures\\NME_forBIRADSdesc\\nonmass_roirecords.db', echo=False) # now on, when adding new cases
        Session = sessionmaker()
        Session.configure(bind=queryengine)  # once engine is available
        session = Session() #instantiate a Session

        # perform query
        ############# by lesion id
        lesion = session.query(localdatabase.Lesion_record, localdatabase.Radiology_record, localdatabase.ROI_record).\
            filter(localdatabase.Radiology_record.lesion_id==localdatabase.Lesion_record.lesion_id).\
            filter(localdatabase.ROI_record.lesion_id==localdatabase.Lesion_record.lesion_id).\
            filter(localdatabase.Lesion_record.lesion_id == str(lesion_id)).options(joinedload_all('*')).all()
        # print results
        if not lesion:
            print "lesion is empty"
            lesion_id = lesion_id+1
            continue
        
        lesion = lesion[0]
        session.close()
        
        MorNMcase = []; cond = [];          
        is_mass = list(lesion.Lesion_record.mass_lesion)
        if(is_mass):
            print "MASS"
            cond = 'mass'
            mass = pd.Series(is_mass[0])
            mass_Case =  pd.Series(mass[0].__dict__)
            print(mass_Case) 
            # decide if it's a mass or nonmass
            MorNMcase = mass_Case
            
        is_nonmass = list(lesion.Lesion_record.nonmass_lesion)
        if(is_nonmass):
            print "NON-MASS"
            cond = 'nonmass'
            nonmass = pd.Series(is_nonmass[0])
            nonmass_Case =  pd.Series(nonmass[0].__dict__)
            print(nonmass_Case) 
            # decide if it's a mass or nonmass
            MorNMcase = nonmass_Case
            
        # first collect only non-masses
        if(cond == 'nonmass'):
            # lesion frame       
            lesion_record = pd.Series(lesion.Lesion_record.__dict__)
            roi_record = pd.Series(lesion.ROI_record.__dict__)
            nmlesion_record = pd.Series(lesion_record['nonmass_lesion'][0].__dict__)       
       
            #lesion_id = lesion_record['lesion_id']
            StudyID = lesion_record['cad_pt_no_txt']
            AccessionN = lesion_record['exam_a_number_txt']
            DynSeries_id = nmlesion_record['DynSeries_id'] 
            roiLabel = roi_record['roi_label']
            zslice = int(roi_record['zslice'])
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
            
            if(lesion_id<=636):
                # to read 4th post-C registered MRI to pre-contrast
                DICOM_path = r'Z:\Breast\DICOMS'         
                print "Reading MRI 4th volume..."
                try:
                    #the output mha:lesionid_patientid_access#_series#@acqusionTime.mha
                    DynSeries_filename = '{}\\{}\\{}'.format(StudyID.zfill(4),AccessionN,DynSeries_nums[4] )
                    DynSeries4th_path = os.path.join(DICOM_path, DynSeries_filename)
                    print "Reading DynSeries4th_images..." 
                    print DynSeries4th_path
                    
                    reader = sitk.ImageSeriesReader()
                    DynSeries4th_UIDs = reader.GetGDCMSeriesIDs(DynSeries4th_path)
                    DynSeries4th_filenames = reader.GetGDCMSeriesFileNames(DynSeries4th_path, DynSeries4th_UIDs[0])
                    reader.SetFileNames(DynSeries4th_filenames)
                    DynSeries4th = reader.Execute()
                    mri4th = sitk.GetArrayFromImage(sitk.Cast(DynSeries4th,sitk.sitkFloat32)) 
                 
                except:
                    print('   failed: locating dynSeries!')
                    lesion_id = lesion_id+1
                    return -1              
            else:
                # to read 4th post-C registered MRI to pre-contrast
                DICOM_path = r'Z:\Cristina\Section3\Breastdata'
                print "Reading MRI 4th volume..."
                try:
                    #the output mha:lesionid_patientid_access#_series#@acqusionTime.mha
                    DynSeries_filename = '{}\\{}\\{}'.format(str(int(StudyID)),AccessionN,DynSeries_nums[4] )
                    DynSeries4th_path = os.path.join(DICOM_path, DynSeries_filename)
                    print "Reading DynSeries4th_images..." 
                    print DynSeries4th_path
                    
                    reader = sitk.ImageSeriesReader()
                    DynSeries4th_UIDs = reader.GetGDCMSeriesIDs(DynSeries4th_path)
                    DynSeries4th_filenames = reader.GetGDCMSeriesFileNames(DynSeries4th_path, DynSeries4th_UIDs[0])
                    reader.SetFileNames(DynSeries4th_filenames)
                    DynSeries4th = reader.Execute()
                    mri4th = sitk.GetArrayFromImage(sitk.Cast(DynSeries4th,sitk.sitkFloat32)) 
                 
                except:
                    print('   failed: locating dynSeries!')
                    lesion_id = lesion_id+1
                    return -1
          
            ###### 3) load DEL and MST graph object into memory
            if(typenxg=='DEL'):
                try:
                    with gzip.open( os.path.join(graphs_path,'{}_{}_{}_FacesTriang_lesion_nxgraph.pklz'.format(str(lesion_id),StudyID.zfill(4),AccessionN)), 'rb') as f:
                        nxGraph = pickle.load(f)
                except:
                    filegraph = glob.glob( os.path.join(graphs_path,'{}_{}_{}_*_FacesTriang_lesion_*'.format(str(lesion_id),StudyID.zfill(4),AccessionN) ))
                    with gzip.open( filegraph[0], 'rb') as f:
                        nxGraph = pickle.load(f)
                nxGraph_name = 'DEL_'+str(lesion_id)
                
            if(typenxg=='MST'):
                try:
                    with gzip.open( os.path.join(graphs_path,'{}_{}_{}_MST_lesion_nxgraph.pklz'.format(str(lesion_id),StudyID.zfill(4),AccessionN)), 'rb') as f:
                        nxGraph = pickle.load(f)
                except:
                    filegraph = glob.glob( os.path.join(graphs_path,'{}_{}_{}_*_MST_*'.format(str(lesion_id),StudyID.zfill(4),AccessionN) ))
                    with gzip.open( filegraph[0], 'rb') as f:
                        nxGraph = pickle.load(f)
                nxGraph_name = 'MST_'+str(lesion_id)
                           
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
            
        # continue
        lesion_id = lesion_id+1

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
    from matplotlib._png import read_png

    ########################################   
    fig_path = r'Z:\Cristina\Section3\NME_DEC\figs'

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
    
    ## plot TSNE
    for i in range(Z_tsne.shape[0]):
        for k in range(len(classes)):
            if str(y_tsne[i])==classes[k]: 
                colori = colors[k] 
        ax1.text(Z_tsne[i, 0], Z_tsne[i, 1], str(y_tsne[i]), color=colori,
                 fontdict={'weight': 'bold', 'size': 8})     
                 
    #############################
    ###### 1) load tsne_id and display (png) 
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
    ###### 2) Examine and plot TSNE with KNN neighbor graphs in a radius of tnse embedding = 0.01
    #############################    
    pdNN = pd.DataFrame({})  
    Z_embedding_tree = spatial.cKDTree(Z_tsne, compact_nodes=True)
    # This finds the index of all points within distance 0.1 of embedded point X_tsne[lesion_id]
    closestd = 0.01
    NN_embedding_indx_list = Z_embedding_tree.query_ball_point(Z_tsne[tsne_id-1], closestd)
    
    while(len(NN_embedding_indx_list)<=6):
        closestd+=0.005
        NN_embedding_indx_list = Z_embedding_tree.query_ball_point(Z_tsne[tsne_id-1], closestd)

    NN_embedding_indx = [knn for knn in NN_embedding_indx_list if knn != tsne_id-1]
    k_nn = min(6,len(NN_embedding_indx))
    
    # plot knn embedded poitns
    for k in range(k_nn):
        k_nn_roid_indx = NN_embedding_indx[k] # finds indices from 0-614, but roi_id => 1-615
            
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
        
        knn_lesion_id = lesion_record['lesion_id']
        StudyID = lesion_record['cad_pt_no_txt']
        AccessionN = lesion_record['exam_a_number_txt']    
        roiLabel = roi_record['roi_label']
        roi_diagnosis = roi_record['roi_diagnosis']
        roi_BIRADS = lesion_record['BIRADS']
        
        #############################
        ###### 3) plot knn closest neighborhs and plot
        #############################
        # us e ConnectorPatch is useful when you want to connect points in different axes
        conknn = ConnectionPatch(xyA=(0,1), xyB=Z_tsne[k_nn_roid_indx], coordsA='axes fraction', coordsB='data',
                axesA=axes[k], axesB=ax1, arrowstyle="simple",connectionstyle='arc3')
        axes[k].add_artist(conknn) 
        print "Indication by y_tsne: %s, by database: %s = %s" % (y_tsne[k_nn_roid_indx],roi_BIRADS+roiLabel,roi_diagnosis)
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
        figTSNE.savefig( 'results/knnTSNE_roid_{}_lesionid_{}_{}_{}.pdf'.format(tsne_id,str(lesion_id),roi_BIRADS+roiLabel,roi_diagnosis), bbox_inches='tight') 
        plt.close()
   
    return pdNN    
        
def read_nxGwimg_features():
    try:
        import cPickle as pickle
    except:
        import pickle
    import gzip

    ## 1) read in the datasets both all NME (to do pretraining)
    NME_nxgraphs = r'Z:\Cristina\Section3\NME_DEC\imgFeatures\NME_nxgraphs'
    # start by loading nxGdatafeatures
    with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_wFU.pklz'), 'rb') as fin:
        nxGdatafeatures = pickle.load(fin)
        
    with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_dynamic_wFU.pklz'), 'rb') as fin:
        allNMEs_dynamic = pickle.load(fin)
        
    with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_morphology_wFU.pklz'), 'rb') as fin:
        allNMEs_morphology = pickle.load(fin)        
        
    with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_texture_wFU.pklz'), 'rb') as fin:
        allNMEs_texture = pickle.load(fin)
        
    with gzip.open(os.path.join(NME_nxgraphs,'allNMEs_stage1_wFU.pklz'), 'rb') as fin:
        allNMEs_stage1 = pickle.load(fin) 
        
                     
    # to load SERw matrices for all lesions
    with gzip.open(os.path.join(NME_nxgraphs,'SER_edgesw_allNMEs_25binsize_wFU.pklz'), 'rb') as fin:
        alldiscrSERcounts = pickle.load(fin)
    
    # to load discrall_dict dict for all lesions
    with gzip.open(os.path.join(NME_nxgraphs,'discrall_dict_allNMEs_10binsize_wFU.pklz'), 'rb') as fin:
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
            nxGdatafeatures['NME_int_enh'].values,
            nxGdatafeatures['dce_init'].values,
            nxGdatafeatures['dce_delay'].values,
            nxGdatafeatures['curve_type'].values,
            nxGdatafeatures['FUstatus'].values,
            nxGdatafeatures['FUtime'].values]
    
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
            nxGdatafeatures['NME_int_enh'].values,
            nxGdatafeatures['dce_init'].values,
            nxGdatafeatures['dce_delay'].values,
            nxGdatafeatures['curve_type'].values]
    
    print('Loading {} NME filled by BC of size = {}'.format(combX_filledbyBC.shape[0], combX_filledbyBC.shape[1]) )
    print('Loading NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_filledbyBC[0].shape[0])   )

    return combX_allNME, YnxG_allNME, combX_filledbyBC, YnxG_filledbyBC           
    
    
    
def read_onlynxG_features():
    try:
        import cPickle as pickle
    except:
        import pickle
    import gzip

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
    YnxG_allNME = [nxGdatafeatures['roi_id'].values,
            nxGdatafeatures['roi_label'].values,
            nxGdatafeatures['roiBIRADS'].values,
            nxGdatafeatures['NME_dist'].values,
            nxGdatafeatures['NME_int_enh'].values,
            nxGdatafeatures['dce_init'].values,
            nxGdatafeatures['dce_delay'].values,
            nxGdatafeatures['curve_type'].values]
    
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
    YnxG_filledbyBC = [nxGdatafeatures['roi_id'].values, # is lesion_id in database nonmass_roirecord
            nxGdatafeatures['roi_label'].values,
            nxGdatafeatures['roiBIRADS'].values,
            nxGdatafeatures['NME_dist'].values,
            nxGdatafeatures['NME_int_enh'].values,
            nxGdatafeatures['dce_init'].values,
            nxGdatafeatures['dce_delay'].values,
            nxGdatafeatures['curve_type'].values]

    print('Loading {} NME filled by BC of size = {}'.format(combX_filledbyBC.shape[0], combX_filledbyBC.shape[1]) )
    print('Loading NME lables [label,BIRADS,dist,enh] of size = {}'.format(YnxG_filledbyBC[0].shape[0])   )
                           
    return combX_allNME, YnxG_allNME, combX_filledbyBC, YnxG_filledbyBC           



def plot_ROC_kStratcv(data, datalabels):
    ## calculate RFmodel ROC
    # shuffle and split training and test sets
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf.get_n_splits(data, datalabels)
    allAUCS = []
    ally_score = []
    ally_test_int = []        
    for train_index, test_index in skf.split(data, datalabels):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = datalabels[train_index], datalabels[test_index]
        
        # train RF model on stratitied kfold
        RFmodel = RFmodel.fit(X_train, y_train)
        y_score = RFmodel.predict_proba(X_test)
        # make malignant class = 1 (positive), Benigng = -1
        y_test_int = [-1 if l=='B' else 1 for l in y_test]
        # pass y_scores as : array, shape = [n_samples] Target scores, can either be probability estimates of the positive class..
        fpr, tpr, thrsh = roc_curve(y_test_int, y_score[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        allAUCS.append(roc_auc)
        ally_score.append(y_score)
        ally_test_int.append(y_test_int)

    print 'mean all kcv AUC =  %0.2f' % np.mean(allAUCS)
    print allAUCS
    stack_ally_score = np.vstack(([ally_score[i] for i in range(len(ally_score))]))
    stack_ally_test_int = np.hstack(([ally_test_int[i] for i in range(len(ally_test_int))]))
    fpr, tpr, thrsh = roc_curve(stack_ally_test_int, stack_ally_score[:,1], pos_label=1)
    print 'StratifiedKFold pooled held-out AUC =  %0.2f' %  auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='green',
             lw=2, label='StratifiedKFold pooled held-out AUC =  %0.2f' %  auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('StratifiedKFold pooled held-out ROC validation')
    plt.legend(loc="lower right")
    plt.show()
    #figscoresM.savefig(dec_model_load+os.sep+'StratifiedKFold_pooled_AUC_znum{}_numc{}_{}.pdf'.format(znum,num_centers,labeltype), bbox_inches='tight')    


def visualize_nxgraphs_metrics(colorlegend=False):
    '''
    # Construct img dictionary calling visualize_nxgraphs_metrics(lesion_id) per lesion_id
    from utilities import visualize_nxgraphs_metrics
    # to run    
    visualize_nxgraphs_metrics()
    # with color bars
    visualize_nxgraphs_metrics(colorlegend=True)
    '''
    import glob, sys, os
    import six.moves.cPickle as pickle
    import gzip
    import SimpleITK as sitk
    import networkx as nx
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sqlalchemy.orm import sessionmaker, joinedload_all
    from sqlalchemy import create_engine
    from sqlalchemy.ext.declarative import declarative_base
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt 

    ################################### 
    # to visualize graphs and MRI data
    ###################################
    NME_nxgraphs = r'Z:\Cristina\Section3\breast_MR_NME_biological\NMEs_SER_nxgmetrics'
    processed_path = r'Y:\Anthony\Cristina\Section3\breast_MR_NME_pipeline\processed_data'
    processed_NMEs_path = r'Z:\Cristina\Section3\breast_MR_NME_biological\processed_NMEs'
    fig_path = r'Z:\Cristina\Section3\NME_DEC\figs'
    
    # gather other infor such as patient levels
    sys.path.insert(0,'Z:\\Cristina\Section3\\breast_MR_NME_biological')
    import localdatabase
    # configure Session class with desired options
    Session = sessionmaker()
    queryengine = create_engine('sqlite:///Z:\\Cristina\\Section3\\breast_MR_NME_biological\\nonmass_roibiological.db', echo=False) # now on, when adding new cases # for filled by Z:\\Cristina\\Section3\\NME_DEC\\imgFeatures\\NME_forBIRADSdesc\\nonmass_roirecords.db
    Session.configure(bind=queryengine)  # once engine is available
    session = Session() #instantiate a Session

    # to load SERw matrices for all lesions
    with gzip.open(os.path.join(NME_nxgraphs,'nxGdatafeatures_allNMEs_descStats.pklz'), 'rb') as fin:
        nxGdatafeatures = pickle.load(fin)
    
    list_lesion_ids = nxGdatafeatures['lesion_id'].values
    list_lesion_ids = list_lesion_ids[list_lesion_ids > 422]
    print list_lesion_ids
    
    for lesion_id in list_lesion_ids:
        # perform query
        ############# by lesion id
        try:
            lesion = session.query(localdatabase.Lesion_record, localdatabase.ROI_record).\
                filter(localdatabase.ROI_record.lesion_id==localdatabase.Lesion_record.lesion_id).\
                filter(localdatabase.Lesion_record.lesion_id == str(lesion_id)).options(joinedload_all('*')).all()
            # print results
        except:
            print "lesion is empty"
            lesion_id = lesion_id+1
            continue
        
        print "lesion id={}".format(lesion_id)
        lesion = lesion[0]    
        # lesion frame       
        lesion_record = pd.Series(lesion.Lesion_record.__dict__)
        NME_record = pd.Series(lesion_record.nonmass_lesion[0].__dict__)
        roi_record = pd.Series(lesion.ROI_record.__dict__)
         
        #lesion_id = lesion_record['lesion_id']
        StudyID = lesion_record['cad_pt_no_txt']
        AccessionN = lesion_record['exam_a_number_txt']
        roi_id = roi_record['roi_id']
        roiLabel = roi_record['roi_label']
        zslice = int(roi_record['zslice'])
        p1 = roi_record['patch_diag1']
        patch_diag1 = p1[p1.find("(")+1:p1.find(")")].split(',')
        patch_diag1 = [float(p) for p in patch_diag1]
        p2 = roi_record['patch_diag2']
        patch_diag2 = p2[p2.find("(")+1:p2.find(")")].split(',')
        patch_diag2 = [float(p) for p in patch_diag2]    
        ext_x = [int(ex) for ex in [np.min([patch_diag1[0],patch_diag2[0]])-20,np.max([patch_diag1[0],patch_diag2[0]])+20] ] 
        ext_y = [int(ey) for ey in [np.min([patch_diag1[1],patch_diag2[1]])-20,np.max([patch_diag1[1],patch_diag2[1]])+20] ] 
    
        ###### 2) Accesing mc images, prob maps, gt_lesions and breast masks
        DynSeries_id = NME_record['DynSeries_id']
        precontrast_id = int(DynSeries_id) 
        DynSeries_nums = [str(n) for n in range(precontrast_id,precontrast_id+5)]

        # to read 4th post-C registered MRI to pre-contrast
        print "Reading MRI 4th volume..."
        try:
            DynSeries_filename = '{}_{}_{}'.format(StudyID.zfill(4),AccessionN,DynSeries_nums[-1] )
            print os.path.join(processed_path,DynSeries_filename+'@*')
            
            #write log if mha file not exist             
            glob_result = glob.glob(os.path.join(processed_path,DynSeries_filename+'@*')) #'*':do not to know the exactly acquistion time
            if glob_result != []:
                filename = glob_result[0] 
                print filename
               
            # read Volumnes
            mriVolDICOM = sitk.ReadImage(filename)
            mri4th = sitk.GetArrayFromImage(sitk.Cast(mriVolDICOM,sitk.sitkFloat32)) 
        except:
            print('   failed: locating MRI 4th volume!')
            
        ###### 3) load graph object into memory
        with gzip.open( os.path.join(processed_NMEs_path,'{}_{}_{}_{}_lesion_nxgraph.pklz'.format(str(roi_id),StudyID.zfill(4),AccessionN,str(roiLabel))), 'rb') as f:
            nxGraph = pickle.load(f)
            
        nxGraph_name = '{}_{}'.format(lesion_id,roiLabel)
           
        ###### 4) plot MRI + graph
        # The triangles in parameter space determine which x, y, z points are connected by an edge
        fig, ax = plt.subplots(dpi=200)   
        # show MRI slice 
        ax.imshow(mri4th[zslice,:,:], cmap=plt.cm.gray)
        ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
         # draw
        nodeweights = [d['weight'] for (u,v,d) in nxGraph.edges(data=True)]
        pos = np.asarray([p['pos'] for (u,p) in nxGraph.nodes(data=True)])
    
        nxg = nx.draw_networkx_edges(nxGraph, pos, ax=ax, edge_color=nodeweights, edge_cmap=plt.cm.inferno, 
                                 width=1.5)
        ax.set_adjustable('box-forced')
        ax.get_xaxis().set_visible(False)                             
        ax.get_yaxis().set_visible(False)
            
        # add color legend
        if(colorlegend):
            v = np.linspace(min(nodeweights), max(nodeweights), 10, endpoint=True)     
            divider = make_axes_locatable(ax)
            caxEdges = divider.append_axes("right", size="9%", pad=0.05)
            plt.colorbar(nxg, cax=caxEdges, ticks=v) 
    
        # save
        fig.savefig( os.path.join(fig_path,nxGraph_name+'.png'), bbox_inches='tight')    
        plt.close()
        
        #################################### plot MRI + metrics
        # DEGREE
        fig, ax = plt.subplots(dpi=200)   
        # show MRI slice 
        ax.imshow(mri4th[zslice,:,:], cmap=plt.cm.gray)
        ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
        Degree = nx.degree_centrality(nxGraph) 
        Dvalues = np.asarray([Degree.get(node) for node in nxGraph.nodes()])
        v = np.linspace(min(Dvalues), max(Dvalues), 10, endpoint=True) 
        nxg = nx.draw_networkx_nodes(nxGraph, pos, ax=ax, node_color=Dvalues, cmap=plt.cm.jet,  
                         node_vmin=np.min(Dvalues), node_vmax=np.max(Dvalues),
                         linewidths=0.0, with_labels=False, node_size=10)
        nx.draw_networkx_edges(nxGraph, pos, ax=ax,  width=0.5)
        ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
        ax.set_adjustable('box-forced')
        ax.get_xaxis().set_visible(False)                             
        ax.get_yaxis().set_visible(False)
        # add color legend
        if(colorlegend):
            divider = make_axes_locatable(ax)
            caxEdges = divider.append_axes("right", size="9%", pad=0.05)
            plt.colorbar(nxg, cax=caxEdges, ticks=v) 
        # save
        fig.savefig( os.path.join(fig_path,nxGraph_name+'_Degree.png'), bbox_inches='tight')    
        plt.close()  
        
        # Betweenness
        fig, ax = plt.subplots(dpi=200)   
        # show MRI slice 
        ax.imshow(mri4th[zslice,:,:], cmap=plt.cm.gray)
        ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
        Betweenness = nx.betweenness_centrality(nxGraph)
        Betweennessvalues = np.asarray([Betweenness.get(node) for node in nxGraph.nodes()])
        v = np.linspace(min(Betweennessvalues), max(Betweennessvalues), 10, endpoint=True) 
        nxg = nx.draw_networkx_nodes(nxGraph, pos, ax=ax, node_color=Betweennessvalues, cmap=plt.cm.jet,  
                         node_vmin=np.min(Betweennessvalues), node_vmax=np.max(Betweennessvalues),
                         linewidths=0.0, with_labels=False, node_size=10)
        nx.draw_networkx_edges(nxGraph, pos, ax=ax,  width=0.5)
        ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
        ax.set_adjustable('box-forced')
        ax.get_xaxis().set_visible(False)                             
        ax.get_yaxis().set_visible(False)
        # add color legend
        if(colorlegend):
            divider = make_axes_locatable(ax)
            caxEdges = divider.append_axes("right", size="9%", pad=0.05)
            plt.colorbar(nxg, cax=caxEdges, ticks=v) 
        # save
        fig.savefig( os.path.join(fig_path,nxGraph_name+'_Betweenness.png'), bbox_inches='tight')    
        plt.close() 
        
        # closeness
        fig, ax = plt.subplots(dpi=200)   
        # show MRI slice 
        ax.imshow(mri4th[zslice,:,:], cmap=plt.cm.gray)
        ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
        closeness = nx.closeness_centrality(nxGraph)
        closeness_values = np.asarray([closeness.get(node) for node in nxGraph.nodes()])
        v = np.linspace(min(closeness_values), max(closeness_values), 10, endpoint=True) 
        nxg = nx.draw_networkx_nodes(nxGraph, pos, ax=ax, node_color=closeness_values, cmap=plt.cm.jet,  
                         node_vmin=np.min(closeness_values), node_vmax=np.max(closeness_values),
                         linewidths=0.0, with_labels=False, node_size=10)
        nx.draw_networkx_edges(nxGraph, pos, ax=ax,  width=0.5)
        ax.axis((ext_y[0], ext_y[1], ext_x[1], ext_x[0]))
        ax.set_adjustable('box-forced')
        ax.get_xaxis().set_visible(False)                             
        ax.get_yaxis().set_visible(False)
        # add color legend
        if(colorlegend):
            divider = make_axes_locatable(ax)
            caxEdges = divider.append_axes("right", size="9%", pad=0.05)
            plt.colorbar(nxg, cax=caxEdges, ticks=v) 
        # save
        fig.savefig( os.path.join(fig_path,nxGraph_name+'_Closeness.png'), bbox_inches='tight')    
        plt.close() 
        
        # continue
    return