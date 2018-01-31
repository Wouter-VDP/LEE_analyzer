
# coding: utf-8

# # LEE Analyzer notebook

# ## Imports & Constants

import math
import glob
import time
import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from root_numpy import list_branches
from root_pandas import read_root

pd.options.display.max_columns = 999
gr      = 1.618
nan     = -2147483648

mass_p= 0.93827 #GeV
mass_e= 0.00511 #GeV

# LAr EM showers
R_moliere =  9.5 # cm
X_o       = 13.9 # cm
E_c       = 0.035# GeV


# ## Definitions


# Dictionary translating the category field in the root tree
categories={1: 'Cosmic',
            2: 'e CC',
            3: 'mu CC',
            4: 'NC',
            5: 'Dirt',
            6: 'Data',
            7: 'Mixed',
            0: 'Other'}

# Fiducial volume borders in x,y,z:
fid_arr= [[10,10],[20,20],[10,50]]

# Minimum reconstructable energies:
min_e = 0.02+mass_e # 20MeV
min_p = 0.04+mass_p # 40MeV

# list ROOT files
#filelist = glob.glob('/home/wouter/Public/*/*.root')
filelist = glob.glob('/run/user/1000/gvfs/sftp:host=uboonegpvm02.fnal.gov,user=wvdp/pnfs/uboone/scratch/users/wvdp/v06_26_01_10/bnb_nu_cosmic_1e0p/*/*.root')

# List of fields in the ROOT tree you want to include
columns = ['n_tracks','n_showers','vx','vy','vz','category',
           'true_vx','true_vy','true_vz','nu_pdg','interaction_type',
           'true_vx_sce','true_vy_sce','true_vz_sce','nu_E',
           'true_shower_x_sce','true_shower_y_sce','true_shower_z_sce','true_shower_pdg',
           'nu_daughters_pdg','nu_daughters_E',
           'nu_daughters_endx','nu_daughters_endy',"nu_daughters_endz",
           'nu_daughters_px','nu_daughters_py','nu_daughters_pz',
           'nu_track_ids','nu_shower_ids','nu_shower_daughters','nu_track_daughters',
           'flash_PE','flash_time',
           'shower_dir_x','shower_dir_y','shower_dir_z',
           "shower_start_x","shower_start_y","shower_start_z",
           'shower_open_angle','shower_length',
           "shower_energy","track_energy_dedx","track_energy_hits",
           "track_dir_x","track_dir_y","track_dir_z",
           "track_start_x","track_start_y","track_start_z",
           'track_end_x','track_end_y','track_end_z',
           'predict_p','predict_mu','predict_pi','predict_em','predict_cos',
           'shower_pca','track_pca','track_nhits','shower_nhits',
           'matched_showers','matched_tracks','matched_tracks_energy','matched_showers_energy',
           'shower_dEdx_hits','track_dEdx_hits'
          ]

print(filelist)


# ## Functions


# Return true if the point is in the TPC with a tolerance.
def CheckBorderTPC(x,y,z,tolerance=0):
    detectorx   =256.35     # In cm
    detectory   =116.5      # Symmetric around 0     
    detectorz   =1036.8
    d=tolerance # border tolerance
    if (0+d) < x < (detectorx-d):
            if (-detectory+d)< y < (detectory-d):
                    if (0+d) < z < (detectorz-d):
                        return True
    return False

def CheckBorderFid(x,y,z,arr):
    detectorx   =256.35     # In cm
    detectory   =116.5      # Symmetric around 0     
    detectorz   =1036.8
    if (0+arr[0][0]) < x < (detectorx-arr[0][1]):
            if (-detectory+arr[1][0])< y < (detectory-arr[1][1]):
                    if (0+arr[2][0]) < z < (detectorz-arr[2][1]):
                        return True
    return False


# Return the angle between two numpy vectors.
def anglevec(v1,v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# Verifies if the event passed the flash precuts.
def flash_precut(row):
    #Flash precuts
    t_start = 3.2
    t_end   = t_start+1.6
    min_PE  = 50

    for time,PE in zip(row['flash_time'],row['flash_PE']):
        if time>t_start and time<t_end and PE>min_PE:
            return True
    return False

# Sum reconstructed energy
def CalcRecoE(row):
    reco_energy = [0,0,0,0]
    total_nhits = [0,0,0]
    
    for tr_hits,tr_dedx,nhits in zip(row['track_energy_hits'],row['track_energy_dedx'],row["track_nhits"]):
        reco_energy[:3]+=tr_hits  
        reco_energy[-1]+=tr_dedx 
        total_nhits+=nhits
        
    for shower,nhits in zip(row['shower_energy'],row["shower_nhits"]):
        reco_energy[:3]+=shower
        reco_energy[-1]+=shower[2] #4th entry uses plane 2 hits for showers and dedx for tracks
        total_nhits+=nhits
        
    return pd.Series({"reconstructed_energy": reco_energy, "total_nhits": total_nhits})

# Definition 
def true_thresholds(row):
    passed_e=False
    passed_p=False
    for pdg,E in zip(row["nu_daughters_pdg"],row["nu_daughters_E"]):
        if pdg==11 and E>min_e:
            passed_e=True
        if pdg==2212 and E>min_p:
            passed_p=True
    return passed_e and passed_p

# Formatting
def sciNot(x):
    x=float(x)
    return "{:.1f}".format(x)

# Calculates the true end point for electron showers, for 95% of energy
def showerTrueEnd(row):
    i_daughter = np.argwhere(row['nu_daughters_pdg']==11)
    i_shower   = np.argwhere(row['true_shower_pdg']==11)
    if len(i_daughter[0])>1:
        print('More than 1 true electron daughter')
    if len(i_shower[0])  >1:
        print('More than 1 true electron shower')
    i_daughter = i_daughter[0][0]
    i_shower   = i_shower[0][0]
    
    E_ratio = (row['nu_daughters_E'])[i_daughter]/E_c
    t_max = np.log(E_ratio)-1.0
    length = (t_max+0.08*18+9.0)*X_o
    
    direction = np.array([row['nu_daughters_px'][i_daughter],row['nu_daughters_py'][i_daughter],row['nu_daughters_pz'][i_daughter]])
    true_shower_start = np.array([row['nu_daughters_vx'][i_daughter],row['nu_daughters_vy'][i_daughter],row['nu_daughters_vz'][i_daughter]])
    true_shower_end = true_shower_start+length*direction/np.linalg.norm(direction)
    true_shower_tmax = true_shower_start+t_max*X_o*direction/np.linalg.norm(direction)
    
    return pd.Series({"true_shower_endx": true_shower_end[0], "true_shower_endy": true_shower_end[1], "true_shower_endz": true_shower_end[2],
                      "true_shower_tmax_x": true_shower_tmax[0], "true_shower_tmax_y": true_shower_tmax[1], "true_shower_tmax_z": true_shower_tmax[2]})

def showerRecoEnd(row):
    for x,y,z,px,py,pz,l in zip(row['shower_start_x'],row['shower_start_y'],row['shower_start_z'],row['shower_dir_x'],row['shower_dir_y'],row['shower_dir_z'],row['shower_length']):
        start=np.array([x,y,z])
        direc=np.array([px,py,pz])
        end=start+l*direc/np.linalg.norm(direc)
        if not CheckBorderTPC(*end):
            return False
    return True

# Calculates the percentage of sps inside the fiducial volume and the percentage of deposited collection plane charge
def containedPercent(row):
    n1=0
    n2=0
    d1=0.1
    d2=0.1
    
    for x,y,z,q in zip(row['shower_sp_x'],row['shower_sp_y'],row['shower_sp_z'],row['shower_sp_int']):
        d1+=1
        if q>0:
            d2+=q
        if CheckBorderFid(x,y,z,fid_arr):
            n1+=1
            if q>0:
                n2+=q

    return pd.Series({"shower_containment_h": n1/d1, "shower_containment_q": n2/d2 })

def add_columns(df_shower):
    # Calculate the true end point of electron showers
    df_shower[['true_shower_endx','true_shower_endy','true_shower_endz','true_shower_tmax_x','true_shower_tmax_y','true_shower_tmax_z']]=df_shower.apply(showerTrueEnd,axis=1)
    # Calculate true containment of electron showers
    df_shower['e_contained']=df_shower.apply(lambda x: CheckBorderTPC(x['true_shower_endx'],x['true_shower_endy'],x['true_shower_endz']), axis=1)
    df_shower['e_contained_tmax']=df_shower.apply(lambda x: CheckBorderTPC(x['true_shower_tmax_x'],x['true_shower_tmax_y'],x['true_shower_tmax_z']), axis=1)
    # Calculate reco containment of showers spacepoints
    df_shower[['shower_containment_h','shower_containment_q']]=df_shower[['shower_sp_x','shower_sp_y','shower_sp_z','shower_sp_int']].apply(containedPercent,axis=1)
     # Calculate reco containment of shower lengths
    df_shower['reco_length_containment']=df_shower.apply(showerRecoEnd,axis=1)
    return df_shower


# Load root files into a dataframe or dictionary of dataframes.

def loadData(filelist,                  # List of input files
             columns,                   # List of fields (columns) to include in the dataframe  
             maxf=4,                    # Maximum number of files to loop over
             contained=True,            # Is the true neutrino interaction in a defined volume?
             truecontains=[11,2212],    # Require the true neutrino interaction to have the following daughters.
             LEE_denom=True,            # 1 proton of at least 40MeV, 1 electron of at least 20MeV, true vertex inside: x[10,10],y[20,20],z[10,50]
             flashpassed=True,          # Output only rows with a candidate passing the optical selection
             passed=True,               # Output only rows with a candidate passing the selection
             returnpure=True ):         # Output only rows with a candidate passing that is within 5cm and not a cosmic.
                     
    columns_req = ['true_vx','true_vy','true_vz',
                   'nu_daughters_vx','nu_daughters_vy','nu_daughters_vz',
                   'flash_time','flash_PE',
                   'nu_daughters_pdg','nu_daughters_E','nu_daughters_px','nu_daughters_py','nu_daughters_pz',
                   'flash_passed','passed','category','distance',
                   'track_energy_hits','shower_energy','track_energy_dedx','shower_nhits','track_nhits',
                   'true_shower_x_sce','true_shower_x_sce',"true_shower_x_sce","true_shower_pdg",
                   'shower_start_x','shower_start_y','shower_start_z',
                   'shower_sp_x','shower_sp_y','shower_sp_z','shower_sp_int',
                   'shower_length',"shower_dir_x","shower_dir_y","shower_dir_z"
                  ]
    columns_remove = [item for item in columns_req if item not in columns]
    
    global categories
    global fid_arr
    chunks=[]
    columns_all = list(set(columns) | set(columns_req))
    
    entries=0
    entries_contained=0
    entries_truecontains=[0]*len(truecontains)
    entries_LEEdef=0
    entries_valid_flash=0
    entries_flashpassed=0
    entries_passed=0
    entries_noncosmic=0
    entries_pure=0
    entries_final=0
    nfiles=len(filelist)
    if maxf<nfiles:
        nfiles=maxf
    
    print
    #print(columns_remove)
    print ('Start to load entries from',nfiles,'files.\n')
    start_time = time.time()
    
    progress=0
    for i_f,fname in enumerate(filelist[:nfiles]):
        print(fname)
        if(i_f % math.ceil(nfiles/10)==0 ):
            print ('Progress:',progress,'%.')
            progress+=10
        
        if (os.stat(fname).st_size)<50000:
            print('File',fname,'was corrupt. Size:',os.stat(fname).st_size/1000, 'kb, skipping to next file.')
            continue
        
        dftemp=read_root(fname,"robertoana/pandoratree",columns=columns_all)
        
        entries+=len(dftemp.index)
        
        if contained:
            dftemp=dftemp[dftemp.apply(lambda x: CheckBorderTPC(x['true_vx'],x['true_vy'],x['true_vz']), axis=1)]
            entries_contained+=len(dftemp.index)
            
        for i,pdg in enumerate(truecontains):
            dftemp = dftemp[dftemp["nu_daughters_pdg"].apply(lambda x: pdg in x)]
            entries_truecontains[i]+=len(dftemp.index)
            
            
        if LEE_denom:
            dftemp = dftemp[dftemp.apply(lambda x: CheckBorderFid(x['true_vx'],x['true_vy'],x['true_vz'],fid_arr), axis=1)]
            dftemp = dftemp[dftemp[["nu_daughters_pdg","nu_daughters_E"]].apply(true_thresholds, axis=1)]
            entries_LEEdef+=len(dftemp.index)
        
        dftemp = dftemp[dftemp.apply(flash_precut, axis=1)]
        entries_valid_flash+=len(dftemp.index)
            
        if flashpassed:
            dftemp = dftemp[ dftemp['flash_passed'].apply(lambda x: not np.all(x==-1))]
            entries_flashpassed+=len(dftemp.index)
            
        if passed:
            dftemp=dftemp[dftemp['passed']==1]
            entries_passed+= len(dftemp.index)
            entries_noncosmic+= len(dftemp[dftemp['category'] !=1 ].index)
            
            entries_pure+= len(dftemp[dftemp['distance']<5].index)
            
        if returnpure:
            dftemp = dftemp[ (dftemp['category'] !=1) & (dftemp['distance']<5) ]
            
        entries_final+=len(dftemp.index)
        
        dftemp["category"]=dftemp["category"].map(categories)        
        dftemp[['reconstructed_energy','total_nhits']]=dftemp[["track_energy_hits","shower_energy",'track_energy_dedx',"shower_nhits","track_nhits"]].apply(CalcRecoE,axis=1)
        dftemp.drop(['true_vx','true_vy','true_vz'],axis=1,inplace=True)
        
        # Do calculations with the small dataframe
        dftemp = add_columns(dftemp)
        
        # Remove columns from the small dataframe
        dftemp.drop(columns_remove,axis=1,inplace=True)
        
        chunks.append(dftemp)
    print('\nSummary:')
    print (entries,'entries were loaded from',nfiles,'files.')
    print (entries_contained, 'entries with true vertex in TPC.')
    for nr,pdg in zip(entries_truecontains,truecontains):
        print (nr, 'entries with a pdg',pdg,'particle.')
    print (entries_LEEdef, 'entries with the LEE signal definition.')
    print (entries_valid_flash, 'entries with a valid flash.')
    print (entries_flashpassed, 'entries passing the optical selection.')
    print (entries_passed, 'entries passing the selection.')
    print ('Category efficiency :', entries_noncosmic,'/', entries_LEEdef,'(',sciNot(entries_noncosmic/entries_LEEdef*100),'%)','signal events passed and was not a cosmic.')
    print ('Closeness purity :', entries_pure,'/', entries_passed,'(',sciNot(entries_pure/entries_passed*100),'%)','of passed events is within 5cm from true_sce vertex.')
    print (entries_final,'entries in the final dataframe.\n')
    
    print('Concatenating the dataframes')
    df = pd.concat(chunks,ignore_index=True,copy=False) 
    end_time = time.time()
    print('Loading took ',sciNot(end_time-start_time),' seconds.')
    return df


# ## Load dataframe and save to Pickle


df = loadData(filelist,                 # List of input files
             columns,                   # List of fields (columns) to include in the dataframe  
             maxf=400,                  # Maximum number of files to loop over
             contained=True,            # Is the true neutrino interaction in a defined volume?
             truecontains=[11,2212],    # Require the true neutrino interaction to have the following daughters.
             LEE_denom=True,            # 1 proton of at least 40MeV, 1 electron of at least 20MeV, true vertex inside: x[10,10],y[20,20],z[10,50]
             flashpassed=True,          # Output only rows with a candidate passing the optical selection
             passed=True,               # Output only rows with a candidate passing the selection
             returnpure=True )          # Output only rows with a candidate passing that is within 5cm and not a cosmic.

df.to_pickle('Input/nue_cosmic_passed_LEEdef.pckl')




