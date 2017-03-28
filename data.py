import os,sys 
import numpy as np
from sklearn.datasets import fetch_mldata
import six.moves.cPickle as pickle
import gzip

info_path = 'Z://Cristina//Section3//breast_MR_NME_pipeline' # os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path = [info_path] + sys.path
from query_localdatabase import *

def get_mnist():
    np.random.seed(1234) # set seed for deterministic ordering
    data_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    data_path = os.path.join(data_path, 'datasets')
    if not os.path.exists(data_path):
        os.mkdir(data_path)        
    print"saving mnist to %s..."%data_path
    mnist = fetch_mldata('MNIST original', data_home=data_path)
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p].astype(np.float32)*0.02
    Y = mnist.target[p]
    return X, Y
    
    
def get_serw():
    np.random.seed(1234) # set seed for deterministic ordering
    data_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    data_path = os.path.join(data_path, 'datasets')
    with gzip.open(os.path.join(data_path,'allNMEs_25binsizenormSERcounts_probC_x.pklz'), 'rb') as fu:
        normSERcounts = pickle.load(fu)
    
    with gzip.open(os.path.join(data_path,'allNMEs_25binsizedatanormSERcounts_probC_x.pklz'), 'rb') as f:
        datanormSERcounts = pickle.load(f)
          
    print"loading discretized SER bins %s..."%data_path
    # set up some parameters and define labels
    X = np.asarray(normSERcounts)
    Y =  datanormSERcounts['class'].values
    print"X is mxn-dimensional with m = %i discretized SER bins" % X.shape[1]
    print"X is mxn-dimensional with n = %i cases" % X.shape[0]    
    return X, Y
    
    
def get_nxGfeatures():
    data_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    data_path = os.path.join(data_path, 'datasets')
    with gzip.open(os.path.join(data_path,'nxGdatafeatures_allNMEs_10binsize.pklz'), 'rb') as fu:
        nxGdatafeatures = pickle.load(fu)

    with gzip.open(os.path.join(data_path,'nxGdiscdatafeatures_allNMEs_10binsize.pklz'), 'rb') as fu:
        nxGdiscdatafeatures = pickle.load(fu)        
    
    with gzip.open(os.path.join(data_path,'nxGnormfeatures_allNMEs_10binsize.pklz'), 'rb') as fu:
        nxGnormfeatures = pickle.load(fu)
              
    print"loading nx graph features %s..."%data_path
    # set up some parameters and define labels
    X = nxGnormfeatures
    print"nxGdatafeatures is mxn-dimensional with m = %i nx graph features" % X.shape[1]
    print"nxGdatafeatures is mxn-dimensional with n = %i cases" % X.shape[0]
    y =  nxGdatafeatures['roiBIRADS'].values
    y0 =  nxGdatafeatures['classNME'].values
    y1 = nxGdatafeatures['roi_id'].values
    y2 = nxGdatafeatures['nme_dist'].values
    y3 = nxGdatafeatures['nme_int'].values
    y4 = nxGdatafeatures['dce_init'].values
    y5 = nxGdatafeatures['dce_delay'].values
    y6 = nxGdatafeatures['nme_t2si'].values
    Y = [y,y0,y1,y2,y3,y4,y5,y6]
    return X, Y, nxGdatafeatures
    
    
def get_pathologyLabels(YnxG):
    
    ################################### 
    #  get roi info from database  
    ###################################
    # get lesion info from database  
    localdata = Querylocal()
    clusterpts_labels = []
    clusterpts_diagnosis = []
    for roi_id in range(len(YnxG[0])):
        label=[]
        dflesion  = localdata.queryby_roid_wpath(roi_id+1)
        wpath = False
        try:
            path_record = pd.Series(dflesion.gtPathology_record.__dict__)
            roi_record = pd.Series(dflesion.ROI_record.__dict__)
            wpath = True
        except:
            roi_record = pd.Series(dflesion.ROI_record.__dict__)
            
        #lesion_id = lesion_record['lesion_id']
        roiLabel = roi_record['roi_label']
        roi_diagnosis = roi_record['roi_diagnosis']
        if(wpath and roiLabel!='U'):            
            if(path_record['histop_core_biopsy_benign_yn']==r'1'):
                label=['B']
            if(path_record['histop_benign_bst_parenchyma_yn']==r'1' or roi_diagnosis=="BENIGN BREAST TISSUE"):
                label=['BBP']

            # hyperplasias
            if(roi_diagnosis=="USUAL DUCTAL HYPERPLASIA" or 
                roi_diagnosis=="ANGIOMATOUS STROMAL HYPERPLASIA" or 
                roi_diagnosis=="COLUMNAR CELL CHANGE WITH HYPERPLASIA" or
                roi_diagnosis=="FLORID USUAL DUCTAL HYPERPLASIA" or
                roi_diagnosis=="RADIAL SCAR WITH FLORID DUCTAL HYPERPLASIA" or
                roi_diagnosis=="FOCAL USUAL DUCTAL HYPERPLASIA"):
                    label=["UDH"]
            if(roi_diagnosis=="ATYPICAL DUCTAL HYPERPLASIA"):
                label=["ADH"]
            if(roi_diagnosis=="PSEUDOANGIOMATOUS STROMAL HYPERPLASIA"):
                label=["PASH"]
            if(roi_diagnosis=="LOBULAR CARCINOMA IN SITU" or
               roi_diagnosis=="LOBULAR CARCINOMA IN SITU WITH FOCAL INVASION"):
                    label=['ISLC']
                
            # malignants
            if(path_record['histop_tp_isc_ductal_yn']==r'1'):
                label=['ISDC']
                if(path_record['in_situ_nucl_grade_int']):
                    label.append('nucl_grade_'+str(path_record['in_situ_nucl_grade_int']))
            
            if(path_record['histop_tp_ic_yn']==r'1'):
                label=['Invasive']
                if(roi_diagnosis=="INVASIVE DUCTAL CARCINOMA"):
                    label.append("DC")
                
            if(path_record['histop_tp_isc_other_txt']):
                label.append(path_record['histop_tp_isc_other_txt'])
        else:
            label = roiLabel
        
        if(label==[]):
            label = raw_input("label not parsed, enter manually: ")
            
        # append
        print".. adding roi_id=%i, with label=%s, and diagnosis=%s \n"%(roi_id, label, roi_diagnosis)
        clusterpts_labels.append( '_'.join(label) )
        clusterpts_diagnosis.append( roi_diagnosis )

    return clusterpts_labels, clusterpts_diagnosis
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    