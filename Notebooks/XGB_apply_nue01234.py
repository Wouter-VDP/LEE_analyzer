
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from operator import itemgetter

from helpfunction import CheckBorderFixed,CheckBorderTPC
from sklearn.externals import joblib

pd.options.display.max_columns = 999


# In[2]:

# Fiducial volume borders in x,y,z:
fid_arr= [[10,10],[20,20],[10,50]]

min_dedx_hits=3


# In[3]:

def shower_features(row):
    x=row["shower_start_x"]
    y=row["shower_start_y"]
    z=row["shower_start_z"]
    vx=row["vx"]
    vy=row["vy"]
    vz=row["vz"]
    
    shower_dedx_cali = row['shower_dedx_cali']
    shower_dedx = row['shower_dedx']
    shower_dedx_med = shower_dedx*shower_dedx_cali
    
    shower_sh_daughter = True if row['shower_daughter']==1 else False
    shower_tr_daughter = True if row['shower_daughter']==2 else False
    shower_multi_daughter = True if row['shower_daughter']==3 else False
    
    shower_is_tr_daughter = True if row['shower_is_daughter']==2 else False
    shower_is_sh_daughter = True if row['shower_is_daughter']==1 else False
    
    shower_vtx         = np.linalg.norm([x-vx,y-vy,z-vz])
    
    return { 'shower_sh_daughter':shower_sh_daughter,
             'shower_tr_daughter':shower_tr_daughter,
             'shower_multi_daughter':shower_multi_daughter,
             'shower_is_tr_daughter':shower_is_tr_daughter,
             'shower_is_sh_daughter':shower_is_sh_daughter,      
             'shower_vtx':shower_vtx,
             'shower_dedx_med':shower_dedx_med
             }


# In[4]:

def track_features(row):  
    x=row["track_start_x"]
    y=row["track_start_y"]
    z=row["track_start_z"]
    x_e=row["track_end_x"]
    y_e=row["track_end_y"]
    z_e=row["track_end_z"]
    vx=row["vx"]
    vy=row["vy"]
    vz=row["vz"]
    
    track_dedx_cali = row['track_dedx_cali']
    track_dedx = row['track_dedx']
    track_dedx_med = track_dedx*track_dedx_cali
    
    track_length      = np.linalg.norm([x-x_e,y-y_e,z-z_e])
    track_containment = CheckBorderFixed(x_e,y_e,z_e,tolerance=10) 
    track_sh_daughter = True if row['track_daughter']==1 else False
    track_tr_daughter = True if row['track_daughter']==2 else False
    track_multi_daughter = True if row['track_daughter']==3 else False
    
    track_is_tr_daughter = True if row['track_is_daughter']==2 else False
    track_is_sh_daughter = True if row['track_is_daughter']==1 else False
    
    track_vtx         = np.linalg.norm([x-vx,y-vy,z-vz])
    track_vtx_end     = np.linalg.norm([vx-x_e,vy-y_e,vz-z_e])
    
    return { 'track_length':track_length,
             'track_containment':track_containment,
             'track_sh_daughter':track_sh_daughter,
             'track_tr_daughter':track_tr_daughter,
             'track_multi_daughter':track_multi_daughter,
             'track_is_tr_daughter':track_is_tr_daughter,
             'track_is_sh_daughter':track_is_sh_daughter,          
             'track_vtx':track_vtx,
             'track_vtx_end':track_vtx_end,
             'track_dedx_med':track_dedx_med
                     }


# In[5]:

columns_req_track = [  'track_dedx','track_dedx_avg','track_dedx_hits','track_pca',
                       'predict_em','predict_mu', 'predict_cos', 'predict_pi', 'predict_p',
                       'track_start_x', 'track_start_y','track_start_z',
                       'track_end_x', 'track_end_y', 'track_end_z','track_daughter','track_is_daughter',
                       'track_cali','track_dedx_cali','track_hits_ratio','track_maxangle',
                       'matched_tracks','matched_tracks_energy', # Command out for data. not used for training 
                       'track_energy','track_nhits' # NOT USED FOR TRAINING
                      ]

columns_req_shower = [ 'shower_dedx','shower_dedx_avg','shower_dedx_hits','shower_pca','shower_open_angle',
                       'shower_start_x', 'shower_start_y','shower_start_z',
                       'shower_length','shower_daughter','shower_is_daughter',
                       'shower_cali','shower_dedx_cali','shower_hits_ratio','shower_maxangle',
                       'matched_showers', 'matched_showers_energy', # Command out for data. not used for training
                       'shower_energy','shower_nhits'  #NOT USED FOR TRAINING
                      ]


columns_flat = ['vx','vy','vz',"candidate_pdg",'numu_cuts','ccnc','qsqr','theta',
                'bnbweight','reconstructed_energy','flash_PE','flash_time',
                'true_vx_sce','true_vy_sce','true_vz_sce','nu_E','nu_pdg',
                'category','distance','CC_daughter_E','nu_daughters_pdg','nu_daughters_E',
                'shower_containment_q','shower_sp_profile',
                'subrun', 'run','event','nu_daughters_pdg','nu_daughters_E'
               ]

columns_track_XGB = ['track_dedx_med', 'track_dedx_avg', 'track_pca',
                     'predict_em', 'predict_mu', 'predict_cos', 'predict_pi', 'predict_p',
                     'track_length', 'track_vtx', 'track_vtx_end',
                     'track_sh_daughter', 'track_tr_daughter',
                     'track_is_tr_daughter','track_is_sh_daughter',
                     'track_hits_ratio',
                     'track_dedx_hits','track_multi_daughter','track_containment','track_maxangle'
                    ] 

columns_shower_XGB = [ 'shower_dedx_med', 'shower_dedx_avg', 'shower_dedx_hits', 'shower_pca',
                       'shower_open_angle', 'shower_length', 'shower_vtx', 'shower_containment_q','shower_sp_profile', 
                       #'shower_sh_daughter','shower_tr_daughter', 'shower_multi_daughter',
                       'shower_is_sh_daughter','shower_is_tr_daughter','shower_maxangle',
                       'shower_hits_ratio']


columns_all = list(set(columns_req_shower) | set(columns_req_track))+columns_flat


# In[6]:

#df_sample = pd.read_pickle('../Input/bnb/data_bnb.pckl')
#df_sample = pd.read_pickle('../Input/nue/nue_0.pckl')


# In[7]:

sample_list = ['../Input/nue/nue_5.pckl','../Input/nue/nue_6.pckl','../Input/nue/nue_7.pckl',
               '../Input/nue/nue_8.pckl','../Input/nue/nue_9.pckl']

 #             ]
#sample_list = [ '../Input/nue/nue_0.pckl',
#                '../Input/nue/nue_1.pckl',
#                '../Input/nue/nue_2.pckl',
#                '../Input/nue/nue_3.pckl',
#                '../Input/nue/nue_4.pckl']

#sample_list = ['../Input/data/bnb.pckl']


# In[8]:

print(columns_all)


# In[9]:

model_sh_e  = joblib.load('../Input/XGBoost/model_sh_e.pkl')
model_sh_cle  = joblib.load('../Input/XGBoost/model_sh_cle.pkl')
model_sh_mu = joblib.load('../Input/XGBoost/model_sh_mu.pkl')

model_tr_e  = joblib.load('../Input/XGBoost/model_tr_e.pkl')
model_tr_cle  = joblib.load('../Input/XGBoost/model_tr_cle.pkl')
model_tr_mu = joblib.load('../Input/XGBoost/model_tr_mu.pkl')


# In[10]:

def XGBoostClassification(row):
    tr_e, tr_cle, tr_mu = [],[],[]
    sh_e, sh_cle, sh_mu = [],[],[]
    if row['fidvol']:
        if len(row['track_end_y']) > 0:
            tr_e ,tr_cle, tr_mu = TrackClassification(row)
        sh_e, sh_cle, sh_mu = ShowerClassification(row)
    
    return pd.Series({
        'track_electron_score': tr_e,
        'track_closest_electron_score': tr_cle,
        'track_muon_score': tr_mu,
        'shower_electron_score': sh_e,
        'shower_closest_electron_score': sh_cle,
        'shower_muon_score': sh_mu,
    })
    
    
def TrackClassification(row):
    pred_tr_e = []
    pred_tr_mu = []
    pred_tr_cle= []
    # Item with less then 5 hits on the collection plane will not be discarded, they will just not have a score for those objects.
    XGB_input = np.zeros([sum(row['track_dedx_hits']>=min_dedx_hits),len(columns_track_XGB)])
    
    tr_ok=0
    for tr in range(len(row['track_start_y'])):
        if row['track_dedx_hits'][tr]>=min_dedx_hits:
            d_tr={}
            for field in columns_req_track:
                d_tr[field]=row[field][tr]
            for field in ['vx','vy','vz']:
                d_tr[field]=row[field]
            d_tr = {**d_tr,**track_features(d_tr)}
            
            XGB_input[tr_ok] = np.asarray(itemgetter(*columns_track_XGB)(d_tr))
            tr_ok+=1
            
        pred_tr_e = model_tr_e.predict_proba( XGB_input )[:,1] 
        pred_tr_cle = model_tr_cle.predict_proba( XGB_input )[:,1]
        pred_tr_mu = model_tr_mu.predict_proba( XGB_input)[:,1]  
        
    return pred_tr_e, pred_tr_cle, pred_tr_mu
    
    
def ShowerClassification(row):
    pred_sh_e = []
    pred_sh_mu = []
    pred_sh_cle = []
    
    XGB_input = np.zeros([sum(row['shower_dedx_hits']>=min_dedx_hits),len(columns_shower_XGB)])
    
    sh_ok=0
    for sh in range(len(row['shower_start_y'])):
        if row['shower_dedx_hits'][sh]>=min_dedx_hits:
            d_sh={}
            for field in columns_req_shower:
                d_sh[field]=row[field][sh]
            for field in ['vx','vy','vz','shower_containment_q','shower_sp_profile']:
                d_sh[field]=row[field]
            d_sh = {**d_sh,**shower_features(d_sh)}  

            XGB_input[sh_ok] = np.asarray( itemgetter(*columns_shower_XGB)(d_sh)) 
            sh_ok+=1
        
    pred_sh_e =  model_sh_e.predict_proba( XGB_input )[:,1] 
    pred_sh_cle =  model_sh_cle.predict_proba( XGB_input )[:,1] 
    pred_sh_mu=  model_sh_mu.predict_proba( XGB_input )[:,1] 
        
    return pred_sh_e, pred_sh_cle, pred_sh_mu


# In[11]:

columns_final = ['vx','vy','vz','event', 'subrun', 'run','shower_nhits','track_nhits',
                 "candidate_pdg",'numu_cuts','ccnc','qsqr','theta',
                 'bnbweight','reconstructed_energy','flash_PE','flash_time',
                 'true_vx_sce','true_vy_sce','true_vz_sce','nu_E','nu_pdg',
                 'category','distance','true_fidvol','CC_daughter_E','nu_daughters_pdg','nu_daughters_E',
                 'shower_containment_q','shower_sp_profile','shower_energy','track_energy',
                 'shower_electron_score','track_muon_score','shower_muon_score','track_electron_score',
                 'track_closest_electron_score','shower_closest_electron_score','fidvol'
                ]


# In[12]:

df_joined = pd.DataFrame()

for sample in sample_list:
    df_sample = pd.read_pickle(sample)
    #df_sample = df_sample
    #print(df_sample.columns)
    droplist = [x for x  in df_sample.columns if (x not in columns_all)]
    print(len(df_sample.index))
    df_sample.drop(droplist,inplace=True,axis=1)
    #print(df_sample.columns)
    df_sample['true_fidvol'] =df_sample.apply(lambda row: CheckBorderTPC(*row[['true_vx_sce','true_vy_sce','true_vz_sce']],array=fid_arr) ,axis=1)
    df_sample['fidvol'] =df_sample.apply(lambda row: CheckBorderTPC(*row[['vx','vy','vz']],array=fid_arr) ,axis=1)
    df_sample = pd.concat([df_sample, df_sample.apply(XGBoostClassification,axis=1)], axis=1)    
    df_sample = df_sample[columns_final]
    df_joined = pd.concat([df_joined,df_sample],ignore_index=True,copy=False)
    print('done sample')
df_joined.head(10)


# In[13]:

df_joined.to_pickle('../nue56789_xgb.pckl')


# In[14]:

print(len(df_joined.index))
print(df_joined['category'].value_counts())


# In[15]:

df_pure=df_sample[df_sample.apply(lambda x: np.all(np.array(x['track_muon_score'])<0.10) and
                                            np.all(np.array(x['shower_muon_score'])<0.15) and
                                         (np.any(np.array(x['shower_electron_score'])>0.9) or np.any(np.array(x['track_electron_score'])>0.99) ),
                                  axis=1)]


# In[16]:

print(len(df_pure.index))
print(df_pure['category'].value_counts())


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



