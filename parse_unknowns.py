# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 14:22:01 2017

@author: DeepLearning
"""

import sys, os
import string
import datetime
import numpy as np
import pandas as pd
import shutil
import glob

dabatase_loc = r'Z:\Cristina\Section3\NME_DEC\imgFeatures'
# code to automatically download dataset
sys.path = [dabatase_loc] + sys.path
import localdatabase
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'font.size': 8})
import seaborn as sns
import numpy.ma as ma
from skimage.measure import find_contours, approximate_polygon
from scipy.spatial import Delaunay
from matplotlib.collections import LineCollection

from process_extractfeatures import *
import cPickle as pickle
import gzip

print "Executing SQL local connection database..."
###### 1) Querying database2 + biomatrix for clinical, pathology, radiology data
queryengine = create_engine('sqlite:///Z:\\Cristina\\Section3\\NME_DEC\\imgFeatures\\nonmass_roirecords.db', echo=False) # now on, when adding new cases
Session = sessionmaker()
Session.configure(bind=queryengine)  # once engine is available
session = Session() #instantiate a Session

# perform query
############# by lesion id
lesion = session.query(localdatabase.Lesion_record, localdatabase.ROI_record).\
    filter(localdatabase.ROI_record.lesion_id==localdatabase.Lesion_record.lesion_id).\
    order_by(localdatabase.ROI_record.lesion_id.asc())
session.close()

# process through db roi_finding records
for idNME in range(942,lesion.count()):
    # lesion frame
    lesion_ids = lesion[idNME]
    casesFrame = pd.Series(lesion_ids.Lesion_record.__dict__)
    # collect info about roi
    roirecord = pd.Series(lesion_ids.ROI_record.__dict__)
    print 'idNME = {}'.format(idNME)
    print 'roi_id = {}'.format(roirecord['roi_id'])
    print 'roi_label = {}'.format(roirecord['roi_label'])   
    print 'lesion_id = {}'.format(casesFrame['lesion_id'])
    #orig_patho = pd.Series(casesFrame['gtpathology'][0].__dict__)
    
    if(roirecord['roi_label']=='U'):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        roirecord['roi_label']
        roirecord['roi_label']
        roirecord['roi_label']
        
        roirecord['roi_label']
        roirecord['roi_label']
        roirecord['roi_label']
        
                
        