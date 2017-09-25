# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:29:12 2017

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

import database
from dateutil.relativedelta import relativedelta

import pandas as pd
from pandasqt.excepthook import excepthook
# use QtGui from the compat module to take care if correct sip version, etc.
from pandasqt.compat import QtGui
from pandasqt.models.DataFrameModel import DataFrameModel
from pandasqt.views.DataTableView import DataTableWidget
from pandasqt.views._ui import icons_rc


def queryBiomatrix_followups(fStudyID, redateID, FUdate):
    ''' Start by creating engine '''
    bioengine = create_engine("postgresql+psycopg2://biomatrix_ruser_mri_cad:bi0matrix4mricadSTUDY@142.76.29.187/biomatrixdb_raccess_mri_cad")
    # we create the engine
    Sessionbio = sessionmaker()
    Sessionbio.configure(bind=bioengine)  # once engine is available
    sessionbio = Sessionbio() #instantiate a Session
    
    query = sessionbio.query(database.Pt_record, database.Cad_record, database.Exam_record, database.Exam_Finding, database.Procedure, database.Pathology).\
         filter(database.Pt_record.pt_id==database.Cad_record.pt_id).\
         filter(database.Cad_record.pt_id==database.Exam_record.pt_id).\
         filter(database.Exam_record.pt_exam_id==database.Exam_Finding.pt_exam_id).\
         filter(database.Exam_record.pt_id==database.Procedure.pt_id).\
         filter(database.Procedure.pt_procedure_id==database.Pathology.pt_procedure_id).\
         filter(database.Cad_record.cad_pt_no_txt == str(fStudyID)).\
         filter(database.Exam_record.exam_dt_datetime >= redateID).\
         filter(database.Exam_record.exam_dt_datetime <= FUdate).all()
    # pass results
    cad_records = [rec.Cad_record.__dict__ for rec in query]
    exam_records = [rec.Exam_record.__dict__ for rec in query]
    finds_records = [rec.Exam_Finding.__dict__ for rec in query]
    proc_records = [rec.Procedure.__dict__ for rec in query]
    patho_records = [rec.Pathology.__dict__ for rec in query]

    cad = pd.DataFrame.from_records(cad_records)
    exam = pd.DataFrame.from_records(exam_records)
    finds = pd.DataFrame.from_records(finds_records)
    proc = pd.DataFrame.from_records(proc_records)
    patho = pd.DataFrame.from_records(patho_records)
    # append all
    if not cad.empty:
        res = pd.concat([cad, exam, finds,  proc, patho], axis=1)
    else:
        res = pd.DataFrame()
    
    return res, cad, exam, finds, proc, patho


def display_query(selcols):
    ## setup a new empty model
    model1 = DataFrameModel()
    ## setup an application and create a table view widget
    app = QtGui.QApplication([])
    widget1 = DataTableWidget()
    widget1.resize(1600, 800)
    widget1.show()
    ## asign the created model
    widget1.setViewModel(model1)
    ## fill the model with data
    model1.setDataFrame(selcols)
    ## start the app"""
    app.exec_()
    return


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
for idNME in range(1071,lesion.count()):
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
        dfFUdata = pd.DataFrame()
        casesFrame.to_csv(sys.stdout)
        if(int(roirecord['zslice'])+1>45): 
            Fsite='R' 
        else: Fsite='L' #raw_input("Enter R or L for finding site: ")
        print 'Slice # for finding = {}'.format(int(roirecord['zslice'])+1)

        # Query biomatrix for imaging history
        fStudyID = casesFrame['cad_pt_no_txt'] # '0463'
        redateID =  casesFrame['exam_dt_datetime'] # datetime.datetime(2014, 1, 14, 0, 0)
        a_number = casesFrame['exam_a_number_txt']
        dfFU6months = pd.DataFrame(); dfFU1year = pd.DataFrame(); dfFU2year = pd.DataFrame();
       
        ############ Q>A   .
        # subclasify unknowns based on <3months FU, <6months FU, 1yFU, 2yFU -> results: Pathology (B or M) or resolved
        FU2ydate = redateID + relativedelta(years=2, months=1)
        FU1ydate = redateID + relativedelta(years=1)
        FU6mdate = redateID + relativedelta(months=6)
        
        # 1) is there imaging done in FU6mdate time frame?
        print "Querying 6months FU imaging in biomatrix..."
        res6m, cad6m, exam6m, finds6m, proc6m, patho6m = queryBiomatrix_followups(fStudyID, redateID, FU6mdate)
        if not res6m.empty and any([num != a_number for num in res6m['a_number_txt']]):
            arec = [num != a_number for num in res6m['a_number_txt']]
            other_imaging = res6m.iloc[arec]
            # select to display
            dfFU6months = pd.concat([other_imaging['exam_dt_datetime'],
                                  other_imaging['a_number_txt'],
                                  other_imaging['previous_exam_when_int'],
                                  other_imaging['previous_exam_reason_int']], axis=1)
            dfFU6months.columns = ['datetime_FU6m', 'a_number_FU6m','previous_exam_when_FU6m', 'previous_exam_reason_FU6m']
                                  
            # 2) is there same finding see at FU6mdate time frame?     
            selexam6m = exam6m.iloc[arec]
            selfinds6m = finds6m.iloc[arec]
            selproc6m = proc6m.iloc[arec]
            selpatho6m = patho6m.iloc[arec]
            
            print 'Slice # for finding = {}'.format(int(roirecord['zslice'])+1)            
            FUk = []
            for k,lesion_finds in enumerate(selfinds6m['all_comments_txt'].values):   
                if( selfinds6m.iloc[k]['side_int'][0] == Fsite or selfinds6m.iloc[k]['side_int']=='Bilateral'):
                    print k, lesion_finds
                    print "FU lesion was in the {}\n".format(selfinds6m.iloc[k]['side_int'])
                    FUk.append(k) # = raw_input("Enter k id: ")
            
            if(len(FUk)>0):        
                selexam6m = selexam6m.iloc[FUk]
                selfinds6m = selfinds6m.iloc[FUk]
                selproc6m = selproc6m.iloc[FUk]
                selpatho6m = selpatho6m.iloc[FUk]
                dfFU6months = dfFU6months.iloc[FUk]
                
                # what is the status of the FU lesion
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                dfFU6months['eval_as_folup_FU6m'] = selexam6m['sty_indicator_add_eval_as_folup_yn']
                dfFU6months['high_risk_at_FU6m'] = selexam6m['sty_indicator_high_risk_at_yn']
                dfFU6months['rout_screening_obsp_FU6m'] = selexam6m['sty_indicator_rout_screening_obsp_yn']
                dfFU6months['solv_diff_img_yn_FU6m'] = selexam6m['sty_indicator_prob_solv_diff_img_yn']            
                dfFU6months['original_report_txt_FU6m'] = selexam6m['original_report_txt'] 
                # change format for visualization
                dfFU6months['datetime_FU6m'] = dfFU6months['datetime_FU6m'].astype('string')
    
                # find if thers' any new procedures done after FU imginga
                FUproc = []
                for k,proc_date in enumerate(selproc6m['proc_dt_datetime'].values): 
                    if(proc_date>=min(other_imaging['exam_dt_datetime'])):
                        FUproc.append(k)
                        
                if(len(FUproc)>0):
                    dfFU6months['proc_dt_datetime_FU6m'] = selproc6m['proc_dt_datetime']
                    dfFU6months['proc_report_FU6m'] = selproc6m['original_report_txt']
                    dfFU6months['proc_lesion_comments_FU6m']  = selproc6m['lesion_comments_txt'] 
                    dfFU6months['surgical_other_FU6m'] = selproc6m['surgical_other_txt']
                    # change format for visualization
                    dfFU6months['proc_dt_datetime_FU6m'] = dfFU6months['proc_dt_datetime_FU6m'].astype('string')
            
            # change format for visualization
            dfFU6months['a_number_FU6m'] = dfFU6months['a_number_FU6m'].astype('string')
            dfFU6months['previous_exam_when_FU6m'] = dfFU6months['previous_exam_when_FU6m'].astype('string')
            dfFU6months['previous_exam_reason_FU6m'] = dfFU6months['previous_exam_reason_FU6m'].astype('string')            
            # select
            display_query(dfFU6months)
            iloc6m = int(raw_input("Enter FU6months row 0-n: "))
            dfFUdata = pd.DataFrame([{'FU6months':True}]) 
            dfFU6months['FU6mcondition'] = raw_input("Enter FU6mcondition e.g: stable, resolved, benign, malignant: ")
        else:
            # continue to nect period
            print "No follow up 6m ..."
            dfFUdata = pd.DataFrame([{'FU6months':False}])
                                      
        print  "Querying 1year FU imaging in biomatrix..."
        res1y, cad1y, exam1y, finds1y, proc1y, patho1y  = queryBiomatrix_followups(fStudyID, FU6mdate, FU1ydate)
        if not res1y.empty and any([num != a_number for num in res1y['a_number_txt']]):
            arec = [num != a_number for num in res1y['a_number_txt']]
            other_imaging = res1y.iloc[arec]
            # select to display
            dfFU1year = pd.concat([other_imaging['exam_dt_datetime'],
                                  other_imaging['a_number_txt'],
                                  other_imaging['previous_exam_when_int'],
                                  other_imaging['previous_exam_reason_int']], axis=1)
            dfFU1year.columns = ['datetime_FU1y', 'a_number_FU1y', 'previous_exam_when_FU1y', u'previous_exam_reason_FU1y']
            # 2) is there same finding see at FU6mdate time frame?
            selexam1y = exam1y.iloc[arec]
            selfinds1y = finds1y.iloc[arec]
            selproc1y = proc1y.iloc[arec]
            selpatho1y = patho1y.iloc[arec]
            
            print 'Slice # for finding = {}'.format(int(roirecord['zslice'])+1)
            FUk = []
            for k,lesion_finds in enumerate(selfinds1y['all_comments_txt'].values):   
                if( selfinds1y.iloc[k]['side_int'][0] == Fsite or selfinds1y.iloc[k]['side_int']=='Bilateral'):
                    print k, lesion_finds
                    print "FU lesion was in the {}\n".format(selfinds1y.iloc[k]['side_int'])
                    FUk.append(k) # = raw_input("Enter k id: ")
            
            if(len(FUk)>0):        
                selexam1y = selexam1y.iloc[FUk]
                selfinds1y = selfinds1y.iloc[FUk]
                selproc1y = selproc1y.iloc[FUk]
                selpatho1y = selpatho1y.iloc[FUk]
                dfFU1year = dfFU1year.iloc[FUk]
                
                # what is the status of the FU lesion
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                dfFU1year['eval_as_folup_FU1y'] = selexam1y['sty_indicator_add_eval_as_folup_yn']
                dfFU1year['high_risk_at_FU1y'] = selexam1y['sty_indicator_high_risk_at_yn']
                dfFU1year['rout_screening_obsp_FU1y'] = selexam1y['sty_indicator_rout_screening_obsp_yn']
                dfFU1year['solv_diff_img_yn_FU1y'] = selexam1y['sty_indicator_prob_solv_diff_img_yn']            
                dfFU1year['original_report_txt_FU1y'] = selexam1y['original_report_txt']
                # change format for visualization
                dfFU1year['datetime_FU1y'] = dfFU1year['datetime_FU1y'].astype('string')
                
                # find if thers' any new procedures done after FU imginga
                FUproc = []
                for k,proc_date in enumerate(selproc1y['proc_dt_datetime'].values): 
                    if(proc_date>=min(other_imaging['exam_dt_datetime'])):
                        FUproc.append(k)
                        
                if(len(FUproc)>0):     
                    dfFU1year['proc_dt_datetime_FU1y'] = selproc1y['proc_dt_datetime']
                    dfFU1year['proc_report_FU1y'] = selproc1y['original_report_txt']
                    dfFU1year['proc_lesion_comments_FU1y']  = selproc1y['lesion_comments_txt'] 
                    dfFU1year['surgical_other_FU1y'] = selproc1y['surgical_other_txt']
                    # change format for visualization
                    dfFU1year['proc_dt_datetime_FU1y'] = dfFU1year['proc_dt_datetime_FU1y'].astype('string')
    
            # change format for visualization
            dfFU1year['a_number_FU1y'] = dfFU1year['a_number_FU1y'].astype('string')
            dfFU1year['previous_exam_when_FU1y'] = dfFU1year['previous_exam_when_FU1y'].astype('string')
            dfFU1year['previous_exam_reason_FU1y'] = dfFU1year['previous_exam_reason_FU1y'].astype('string')
            # select
            display_query(dfFU1year)
            iloc1y = int(raw_input("Enter FU1year row 0-n: "))
            dfFUdata['FU1year'] = True 
            dfFU1year['FU1ycondition'] = raw_input("Enter FU1ycondition e.g: stable, resolved, benign, malignant: ")
        else:
            # continue to nect period
            print "No follow up 1y ..."
            dfFUdata['FU1year'] = False

        # continue to nect period
        print "Querying 2year FU imaging in biomatrix..."
        res2y, cad2y, exam2y, finds2y, proc2y, patho2y  = queryBiomatrix_followups(fStudyID, FU1ydate, FU2ydate)
        if not res2y.empty and any([num != a_number for num in res2y['a_number_txt']]):
            arec = [num != a_number for num in res2y['a_number_txt']]
            other_imaging = res2y.iloc[arec]
            # select to display
            dfFU2year = pd.concat([other_imaging['exam_dt_datetime'],
                                  other_imaging['a_number_txt'],
                                  other_imaging['previous_exam_when_int'],
                                  other_imaging['previous_exam_reason_int']], axis=1)
            dfFU2year.columns = ['datetime_FU2y', 'a_number_FU2y', 'previous_exam_when_FU2y', u'previous_exam_reason_FU2y']
    
            # 2) is there same finding see at FU6mdate time frame?
            selexam2y = exam2y.iloc[arec]
            selfinds2y = finds2y.iloc[arec]
            selproc2y = proc2y.iloc[arec]
            selpatho2y = patho2y.iloc[arec]
            
            print 'Slice # for finding = {}'.format(int(roirecord['zslice'])+1)            
            FUk = []
            for k,lesion_finds in enumerate(selfinds2y['all_comments_txt'].values):   
                if( selfinds2y.iloc[k]['side_int'][0] == Fsite or selfinds2y.iloc[k]['side_int']=='Bilateral'):
                    print k, lesion_finds
                    print "FU lesion was in the {}\n".format(selfinds2y.iloc[k]['side_int'])
                    FUk.append(k) # = raw_input("Enter k id: ")
            
            if(len(FUk)>0):        
                selexam2y = selexam2y.iloc[FUk]
                selfinds2y = selfinds2y.iloc[FUk]
                selproc2y = selproc2y.iloc[FUk]
                selpatho2y = selpatho2y.iloc[FUk]
                dfFU2year = dfFU2year.iloc[FUk]
                
                # what is the status of the FU lesion
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                dfFU2year['eval_as_folup_FU2y'] = selexam2y['sty_indicator_add_eval_as_folup_yn']
                dfFU2year['high_risk_at_FU2y'] = selexam2y['sty_indicator_high_risk_at_yn']
                dfFU2year['rout_screening_obsp_FU2y'] = selexam2y['sty_indicator_rout_screening_obsp_yn']
                dfFU2year['solv_diff_img_yn_FU2y'] = selexam2y['sty_indicator_prob_solv_diff_img_yn']            
                dfFU2year['original_report_txt_FU2y'] = selexam2y['original_report_txt']
                # change format for visualization
                dfFU2year['datetime_FU2y'] = dfFU2year['datetime_FU2y'].astype('string')
                
                # find if thers' any new procedures done after FU imginga
                FUproc = []
                for k,proc_date in enumerate(selproc2y['proc_dt_datetime'].values): 
                    if(proc_date>=min(other_imaging['exam_dt_datetime'])):
                        FUproc.append(k)
                        
                if(len(FUproc)>0):     
                    dfFU2year['proc_dt_datetime_FU2y'] = selproc2y['proc_dt_datetime']
                    dfFU2year['proc_report_FU2y'] = selproc2y['original_report_txt']
                    dfFU2year['proc_lesion_comments_FU2y']  = selproc2y['lesion_comments_txt'] 
                    dfFU2year['surgical_other_FU2y'] = selproc2y['surgical_other_txt']
                    # change format for visualization
                    dfFU2year['proc_dt_datetime_FU2y'] = dfFU2year['proc_dt_datetime_FU2y'].astype('string')
                    #dfFU2year = dfFU2year.iloc[FUproc]
                    
            # change format for visualization
            dfFU2year['a_number_FU2y'] = dfFU2year['a_number_FU2y'].astype('string')
            dfFU2year['previous_exam_when_FU2y'] = dfFU2year['previous_exam_when_FU2y'].astype('string')
            dfFU2year['previous_exam_reason_FU2y'] = dfFU2year['previous_exam_reason_FU2y'].astype('string')

            # sekect
            display_query(dfFU2year)
            iloc2y = int(raw_input("Enter FU2year row 0-n: "))
            dfFUdata['FU2year'] = True 
            dfFU2year['FU2ycondition'] = raw_input("Enter FU2ycondition e.g: stable, resolved, benign, malignant: ")
  
        else:
            # no folloup found at any time frame     
            print "NO  FU found..."
            dfFUdata['FU2year'] = False
            print dfFUdata
            
        if not dfFU6months.empty:
            keys6mFU = ['datetime_FU6m', 'proc_dt_datetime_FU6m', 'proc_report_FU6m', 'proc_lesion_comments_FU6m', 'surgical_other_FU6m', 'original_report_txt_FU6m']
            for key in keys6mFU:
                if(key in dfFU6months.keys()):
                    if(key=='datetime_FU6m' or key=='proc_dt_datetime_FU6m'):
                        dfFU6months[key] = dfFU6months[key].astype('datetime64')
                    else:
                        dfFU6months[key] = dfFU6months[key].astype('string')
            
        if not dfFU1year.empty:
            keys1yFU = ['datetime_FU1y', 'proc_dt_datetime_FU1y', 'proc_report_FU1y', 'proc_lesion_comments_FU1y', 'surgical_other_FU1y', 'original_report_txt_FU1y']
            for key in keys1yFU:                    
                if(key in dfFU1year.keys()):
                    if(key=='datetime_FU1y' or key=='proc_dt_datetime_FU1y'):
                        dfFU1year[key] = dfFU1year[key].astype('datetime64')
                    else:
                        dfFU1year[key] = dfFU1year[key].astype('string')

        if not dfFU2year.empty:
            keys2yFU = ['datetime_FU2y', 'proc_dt_datetime_FU2y', 'proc_report_FU2y', 'proc_lesion_comments_FU2y', 'surgical_other_FU2y', 'original_report_txt_FU2y']
            for key in keys2yFU:
                if(key in dfFU2year.keys()):
                    if(key=='datetime_FU2y' or key=='proc_dt_datetime_FU2y'):
                        dfFU2year[key] = dfFU2year[key].astype('datetime64')
                    else:
                        dfFU2year[key] = dfFU2year[key].astype('string')
                        
                    
        session = Session() #instantiate a Session
        for roi_dbrec in session.query(localdatabase.Lesion_record, localdatabase.ROI_record).\
                filter(localdatabase.ROI_record.lesion_id==localdatabase.Lesion_record.lesion_id).\
                filter(localdatabase.Lesion_record.lesion_id == str(casesFrame['lesion_id'])).all():
                    # update if there's follow UPS
                    for key, value in dfFUdata.iloc[0].iteritems():
                        #print key, value
                        roi_dbrec.ROI_record.__setattr__(key, value)
                    # update if there's follow up on 6m
                    if not dfFU6months.empty:
                        for key, value in dfFU6months.iloc[iloc6m].iteritems():
                            #print key, value
                            roi_dbrec.ROI_record.__setattr__(key, value)   
                    # update if there's follow up on 1y
                    if not dfFU1year.empty:
                        for key, value in dfFU1year.iloc[iloc1y].iteritems():
                            #print key, value
                            roi_dbrec.ROI_record.__setattr__(key, value)
                    # update if there's follow up on 2y
                    if not dfFU2year.empty:
                        for key, value in dfFU2year.iloc[iloc2y].iteritems():
                            #print key, value
                            roi_dbrec.ROI_record.__setattr__(key, value)
      
        session.commit()
        session.close()
        idNME = idNME+1
        continue
            

    # TODO: 
    ## Plot imshow of roirecord['zslice'] and compare with follow UPS if indicated in RAD REPORT
 


