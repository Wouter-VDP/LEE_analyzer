
# coding: utf-8

# # LEE Analyzer notebook

# This notebook takes in the output directory of the jobs and convert it into a more flat pandas dataframe. 
# Data from the root files will be partially processed to fields that are convenient to plot.
# The resulting dataframe will be pickled.

# ## Imports & Constants

# In[1]:

import math
import glob
import time
import os
import sys
import numpy as np
import pandas as pd
from root_pandas import read_root
from collections import OrderedDict

from helpfunction import safely_reduce_dtype,reduce_mem_usage,sciNot,CheckBorderTPC,CheckBorderFixed


# In[2]:

pd.options.display.max_columns = 999
gr      = 1.618
nan     = -2147483648
min_root_size = 20000 # Skip root files smaller than x bytes

mass_p= 0.93827 #GeV
mass_e= 0.00511 #GeV

# LAr EM showers
R_moliere =  9.5 # cm
X_o       = 13.9 # cm
E_c       = 0.035# GeV


# In[3]:

# Which plane do we want to use:
plane = 2 # collection plane only

# Fiducial volume borders in x,y,z:
fid_arr= [[10,10],[20,20],[10,50]]
# Fiducial volume for the end points of tracks
fid_min = 10
# Percentage cut for shower containment
sh_cont_percent = .75

# Minimum reconstructable energies:
min_e = 0.02+mass_e # 20MeV
min_p = 0.04+mass_p # 40MeV

# Flat columns we want to copy from the original dataframe:
flat_columns = ["event","subrun","run","nu_pdg","nu_E","true_vx_sce","true_vy_sce","true_vz_sce",
                "distance","category","vx","vy","vz","bnbweight","passed","candidate_pdg",'numu_cuts','ccnc','qsqr','theta']

vec_columns = ["shower_open_angle","shower_length","shower_start_x","shower_start_y","shower_start_z",
              "shower_dir_x","shower_dir_y","shower_dir_z","shower_pca",

              "track_start_x","track_start_y","track_start_z","track_end_x","track_end_y","track_end_z",
              "track_dir_x","track_dir_y","track_dir_z","track_pca",
              "predict_em","predict_mu","predict_cos","predict_pi","predict_p",                                   # Katrin's BDT
              "track_pidchi","track_pidchipr","track_pidchika","track_pidchipi","track_pidchimu","track_pida",    # Adam's PID
              "track_res_mean","track_res_std", 

              "true_shower_pdg",
              #"true_shower_x_sce","true_shower_y_sce","true_shower_z_sce","true_shower_depE"
              ]


# ## Select Input Files

# In[4]:

checkana=True

if not checkana:
    # Option 1: no checkana, verify validity yourself and use input directory
    # Can be on gpvms, example: "/run/user/1000/gvfs/sftp:host=uboonegpvm02.fnal.gov,user=wvdp/uboone/data/users/wvdp/v06_26_01_10/data_bnb_a_1e0p/*/*.root"
    # Local will be faster, avoid using pnfs scratch
    inputlist = []
    inputlist += glob.glob('/home/wouter/Templates/Run_March/bnb_nue_cosmic_1s0t/*/*.root')
    #inputlist += glob.glob(gpvmsdir+'/uboone/data/users/wvdp/v06_26_01_12/data_bnb_b_1s0t/*/*.root')
    filelist  = []
    for fname in inputlist:
        if (os.stat(fname).st_size)<min_root_size*80:
                print("File",fname,"was corrupt. Size:",os.stat(fname).st_size/1000, "kb, skipping to next file.")
        else:
            filelist.append(fname)

if checkana:
    # Option 2: After checkana, just pass the filesana list.
    gpvmsdir = '/run/user/1000/gvfs/sftp:host=uboonegpvm02.fnal.gov,user=wvdp'
    filelist = [gpvmsdir+line.rstrip() for line in open(gpvmsdir+'/uboone/data/users/wvdp/book/v06_26_01_12/data_bnb_b_1s0t/filesana.list')]
    filelist+= [gpvmsdir+line.rstrip() for line in open(gpvmsdir+'/uboone/data/users/wvdp/book/v06_26_01_12/data_bnb_a_1s0t/filesana.list')]

print(len(filelist),"valid ROOT files collected.")


# ## Classes

# In[5]:

## TODO: List of lines that are causing warnings:

#indexing-view-versus-copy
#df_new[col]=dftemp[col].apply(safely_reduce_dtype)

#VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
#RuntimeWarning: divide by zero encountered in float_scalars
#RuntimeWarning: invalid value encountered in double_scalars
#return pd.Series({"shower_sp_profile": np.mean(y[:l])/np.mean(y[-l:])})

#RuntimeWarning: invalid value encountered in true_divide
#center/=total_Q

#RuntimeWarning: invalid value encountered in int_scalars
#shower_hits_ratio[sh] = n_spacepoint[plane]/n_cluster[plane]
#track_hits_ratio[tr] = n_spacepoint[plane]/n_cluster[plane]

#RuntimeWarning: overflow encountered in double_scalars
#dedx = [tr_E[0]*tr_cali[0],tr_E[1]*tr_cali[1],tr_E[2]*tr_cali[2],sum(nhits)/this_nhits]


# ## Functions

# ### Help Functions

# In[ ]:




# ### Selection Functions

# In[6]:

# Signal Definition 1e0p 
def true_thresholds_1e0p(row):
    if CheckBorderTPC(*row[["true_vx_sce","true_vy_sce","true_vz_sce"]],array=fid_arr):
        passed_e=False
        for pdg,E in zip(*row[["nu_daughters_pdg","nu_daughters_E"]]):
            if pdg==11 and E>min_e:
                return True
    return False

def KatrinBDT(row):
    if np.all(row["predict_p"]>0.001):
        if np.all(row["predict_mu"]<0.6):
            if np.all(row["predict_cos"]<0.6):
                return True
    return False


# ### Feature functions

# In[7]:

# Verifies if the event passed the flash precuts.
def FlashPrecut(row):
    flash_ok = False
    t_start = 3.2
    t_end   = t_start+1.6
    min_PE  = 50

    for time,PE in zip(*row[["flash_time","flash_PE"]]):
        if time>t_start and time<t_end and PE>min_PE:
            flash_ok = True
    return pd.Series({"flash_precut": flash_ok}) 



# Reconstructed energy using collection plane
def CalcRecoE(row):
    reco_energy = 0
    total_nhits = 0 # sum over all the planes
    shower_nhits = np.zeros(row["n_showers"]) 
    shower_energy = np.zeros(row["n_showers"])
    shower_cali = np.zeros(row["n_showers"])
    shower_hits_ratio = np.zeros(row["n_showers"])
    
    track_nhits = np.zeros(row["n_tracks"])
    track_energy = np.zeros(row["n_tracks"])
    track_cali = np.zeros(row["n_tracks"])
    track_hits_ratio = np.zeros(row["n_tracks"])
    
    for tr, tr_E, n_cluster, n_spacepoint, cali in zip(range(row["n_tracks"]),*row[["track_energy_hits","track_nhits_cluster","track_nhits_spacepoint","track_energy_cali"]] ):
        reco_energy+=tr_E[plane]
        track_nhits[tr] = int(n_cluster[plane])
        track_energy[tr] = tr_E[plane]
        track_cali[tr] = cali[plane]
        track_hits_ratio[tr] = n_spacepoint[plane]/n_cluster[plane]
        total_nhits+=int(sum(n_cluster))
        
    for sh, sh_E, n_cluster, n_spacepoint, cali in zip(range(row["n_showers"]),*row[["shower_energy_hits","shower_nhits_cluster","shower_nhits_spacepoint","shower_energy_cali"]] ):
        reco_energy+=sh_E[plane]
        shower_nhits[sh] = int(n_cluster[plane])
        shower_energy[sh] = sh_E[plane]
        shower_cali[sh] = cali[plane]
        shower_hits_ratio[sh] = n_spacepoint[plane]/n_cluster[plane]
        total_nhits+=int(sum(n_cluster))
        
    return pd.Series({"reconstructed_energy": reco_energy, "total_nhits": total_nhits,
                      "shower_nhits": shower_nhits, "shower_energy": shower_energy, "shower_cali": shower_cali, "shower_hits_ratio" : shower_hits_ratio,
                      "track_nhits": track_nhits, "track_energy": track_energy, "track_cali": track_cali, "track_hits_ratio" : track_hits_ratio})


# Add for every oject the angle it has with it's most opposite object:
def MaxAngle(row):
    track_maxangle = np.zeros(row["n_tracks"])
    shower_maxangle = np.zeros(row["n_showers"])
    
    track_dirs = zip(*row[['track_dir_x','track_dir_y','track_dir_z']])
    shower_dirs = zip(*row[['shower_dir_x','shower_dir_y','shower_dir_z']])
    alldir = list(track_dirs) + list(shower_dirs)
    #print(track_dirs)
    #print(alldir)
    
    for nr,(this_dir_x, this_dir_y, this_dir_z) in enumerate(zip(*row[['track_dir_x','track_dir_y','track_dir_z']])):
        cosine=1
        for dir_x, dir_y, dir_z in alldir:
            u=np.array([this_dir_x,this_dir_y,this_dir_z])
            v=np.array([dir_x,dir_y,dir_z])
            if np.array_equal(u,v):
                continue
            this_cosine = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v)
            if this_cosine < cosine:
                cosine=this_cosine
        track_maxangle[nr]=cosine
        
    for nr,(this_dir_x, this_dir_y, this_dir_z) in enumerate(zip(*row[['shower_dir_x','shower_dir_y','shower_dir_z']])):
        cosine=1
        for dir_x, dir_y, dir_z in alldir:
            u=np.array([this_dir_x,this_dir_y,this_dir_z])
            v=np.array([dir_x,dir_y,dir_z])
            if np.array_equal(u,v):
                continue
            this_cosine = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v)
            if this_cosine < cosine:
                cosine=this_cosine
        shower_maxangle[nr]=cosine
    
    return pd.Series({"track_maxangle": track_maxangle, "shower_maxangle": shower_maxangle })
                   
    

# Add info about the hierarchy, 0= no daughter, 1=has showerdaughter, 2=has trackdaughter, 3=more than 1
def DaughterInfo(row):
    nu_shower_ids, nu_track_ids                = row["nu_shower_ids"],row["nu_track_ids"]
    nu_shower_daughters, nu_track_daughters    = list(row["nu_shower_daughters"]),list(row["nu_track_daughters"])

    showerdaughter = []
    for sh in nu_shower_daughters:
        if len(sh)==0:
            showerdaughter.append(0)
        elif len(sh)==1:
            if sh[0] in nu_shower_ids:
                showerdaughter.append(1)
            elif sh[0] in nu_track_ids:
                showerdaughter.append(2)
        else:
            showerdaughter.append(3)
            #print("Shower had more than one daughter")
            
    trackdaughter = []
    for tr in nu_track_daughters:
        if len(tr)==0:
            trackdaughter.append(0)
        elif len(tr)==1:
            if tr[0] in nu_shower_ids:
                trackdaughter.append(1)
            elif tr[0] in nu_track_ids:
                trackdaughter.append(2)
        else:
            trackdaughter.append(3)
            #print("Track had more than one daughter")
            
    
    nu_shower_daughters     = np.hstack(row["nu_shower_daughters"]) # want them as flat numpy arrays
    if len(nu_track_daughters)>0:
        nu_track_daughters= np.hstack(row["nu_track_daughters"])
    
    if len(nu_shower_daughters)+len(nu_track_daughters) ==0:
        shower_is_daughter=[0]*len(nu_shower_ids)
        track_is_daughter=[0]*len(nu_track_ids)
    else:
        shower_is_daughter = []
        for sh in nu_shower_ids:
            if sh in nu_shower_daughters:
                shower_is_daughter.append(1)
            elif sh in nu_track_daughters:
                shower_is_daughter.append(2)
            else:
                shower_is_daughter.append(0)

        track_is_daughter = []
        for tr in nu_track_ids:
            if tr in nu_shower_daughters:
                track_is_daughter.append(1)
            elif tr in nu_track_daughters:
                track_is_daughter.append(2)
            else:
                track_is_daughter.append(0)
            
    return pd.Series({"shower_daughter": showerdaughter, "track_daughter": trackdaughter,
                      "shower_is_daughter": shower_is_daughter, "track_is_daughter": track_is_daughter,
                     })     
    

# Calculates the true end point for electron showers, for 95% of energy
def ShowerTrueEnd(row):
    if 11 not in row["nu_daughters_pdg"]:
        return pd.Series({"true_shower_endx": -999.0, "true_shower_endy": -999.0, "true_shower_endz": -999.0,
                      "true_shower_tmax_x": -999.0, "true_shower_tmax_y": -999.0, "true_shower_tmax_z": -999.0})
    i_daughter = np.nonzero(row["nu_daughters_pdg"]==11)
    if len(i_daughter[0])>1:
        print("More than 1 true electron daughter")
    i_daughter = i_daughter[0][0]
    
    E_ratio = (row["nu_daughters_E"][i_daughter])/E_c
    t_max = np.log(E_ratio)-1.0
    length = (t_max+0.08*18+9.6)*X_o
    #print("E_ratio",E_ratio,"E",row["nu_daughters_E"][i_daughter]," t_max",t_max,"length",length)
    direction = np.array([row["nu_daughters_px"][i_daughter],row["nu_daughters_py"][i_daughter],row["nu_daughters_pz"][i_daughter]])
    true_shower_start = np.array([row["nu_daughters_vx"][i_daughter],row["nu_daughters_vy"][i_daughter],row["nu_daughters_vz"][i_daughter]])
    true_shower_end = true_shower_start+length*direction/np.linalg.norm(direction)
    true_shower_tmax = true_shower_start+(t_max*X_o)*direction/np.linalg.norm(direction)
    
    return pd.Series({"true_shower_endx": true_shower_end[0], "true_shower_endy": true_shower_end[1], "true_shower_endz": true_shower_end[2],
                      "true_shower_tmax_x": true_shower_tmax[0], "true_shower_tmax_y": true_shower_tmax[1], "true_shower_tmax_z": true_shower_tmax[2]})


def TrueDaughterInfo(row):
    #"nu_daughters_pdg","nu_daughters_E",
    #"nu_daughters_px","nu_daughters_py","nu_daughters_pz",
    #"nu_daughters_endx","nu_daughters_endy","nu_daughters_endz"
    nu_daughters_pdg = []
    nu_daughters_E = []
    for pdg,E in zip(*row[["nu_daughters_pdg", "nu_daughters_E"]]):
        if pdg!=2112 and pdg<9999: #Do not store neutrons and nuclei
            nu_daughters_pdg.append(pdg)
            nu_daughters_E.append(E)
    return pd.Series({"nu_daughters_pdg": np.array(nu_daughters_pdg), "nu_daughters_E": np.array(nu_daughters_E)})

    

# Calculates the percentage of sps inside the fiducial volume and the percentage of deposited collection plane charge
def ContainedRatio(row):
    d=0.01
    n=0.0
    
    for x,y,z,q in zip(*row[["shower_sp_x", "shower_sp_y", "shower_sp_z", "shower_sp_int"]]):
        d+=q
        if CheckBorderTPC(x,y,z,array=fid_arr):
            n+=q

    return pd.Series({"shower_containment_q": n/d}) 



def CC_daughter_E(row):
    CC_daughter_i = np.nonzero(np.in1d(row["nu_daughters_pdg"], [11,-11,13,-13]))[0]
    CC_daughter_E = -1
    if len(CC_daughter_i)>0:
        if len(CC_daughter_i)>2:
            "Multiple electron/muon daughters found!"
        else:      
            CC_daughter_E = row["nu_daughters_E"][CC_daughter_i[0]]
    return pd.Series({"CC_daughter_E": CC_daughter_E})  



# Returns the ratio of collection charge of the first part and the second part of the summed shower.
def ShowerChargeProfile(row):
    x,y,z = row["vx"],row["vy"],row["vz"]
    
    center= np.array([0.0,0.0,0.0])
    total_Q = 0.0
    for sps_x,sps_y,sps_z,sps_int in zip(*row[["shower_sp_x","shower_sp_y","shower_sp_z","shower_sp_int"]]):
        center+=np.array([sps_x,sps_y,sps_z])*sps_int
        total_Q+=sps_int
    center/=total_Q
    norm = (center-np.array([x,y,z])) / np.linalg.norm(center-np.array([x,y,z]))
    
    distance = []
    weights = []
    for sps_x,sps_y,sps_z in zip(*row[["shower_sp_x","shower_sp_y","shower_sp_z"]]):
        if sps_int>0:
            distance.append( np.dot([sps_x-x,sps_y-y,sps_z-z],norm) )
            weights.append(sps_int)
            
    y,x = np.histogram( distance, weights = weights )
    l = len(y)/2
    
    #ratio = np.mean(y[:l])/np.mean(y[-l:]) if np.mean(y[-l:])>0 else -1
    return pd.Series({"shower_sp_profile": np.mean(y[:l])/np.mean(y[-l:])})  


# Returns the dedx and the number of hits it had to compute this.
def CalcDedx(row):
    shower_dedx_hits = np.zeros(row["n_showers"]) 
    shower_dedx      = np.zeros(row["n_showers"])
    shower_dedx_avg  = np.zeros(row["n_showers"]) 
    shower_dedx_cali = np.zeros(row["n_showers"]) 
    
    track_dedx_hits  = np.zeros(row["n_tracks"])
    track_dedx       = np.zeros(row["n_tracks"])
    track_dedx_avg   = np.zeros(row["n_tracks"])
    track_dedx_cali  = np.zeros(row["n_tracks"]) 
    
    
    for tr, tr_E, nhits,tr_cali in zip(range(row["n_tracks"]),*row[["track_dEdx","track_dEdx_hits","track_dQdx_cali"]]):
        this_nhits = len(nhits)
        track_dedx_hits[tr] = this_nhits
        if this_nhits==0:
            track_dedx_avg[tr]  = 0
            track_dedx[tr]      = 0
            track_dedx_cali[tr] = 0
        else:
            dedx = [tr_E[0]*tr_cali[0],tr_E[1]*tr_cali[1],tr_E[2]*tr_cali[2],sum(nhits)/this_nhits]
            #just force the code to take the best dedx if it would be an electron.
            track_dedx_avg[tr]  = dedx[np.argmin(abs(np.array(dedx)-1.95))]
            track_dedx[tr]      = tr_E[plane]
            track_dedx_cali[tr] = tr_cali[plane]
        
    for sh, sh_E, nhits,sh_cali in zip(range(row["n_showers"]),*row[["shower_dEdx","shower_dEdx_hits","shower_dQdx_cali"]]):
        this_nhits = len(nhits)
        shower_dedx_hits[sh] = this_nhits
        if this_nhits==0:
            shower_dedx_avg[sh]  = 0
            shower_dedx[sh]      = 0
            shower_dedx_cali[sh] = 0
        else:
            dedx = [sh_E[0]*sh_cali[0],sh_E[1]*sh_cali[1],sh_E[2]*sh_cali[2],sum(nhits)/this_nhits]
            #just force the code to take the best dedx if it would be an electron.
            shower_dedx_avg[sh]  = dedx[np.argmin(abs(np.array(dedx)-1.95))]
            shower_dedx[sh]      = sh_E[plane]
            shower_dedx_cali[sh] = sh_cali[plane]
    
    return pd.Series({"shower_dedx_hits": shower_dedx_hits, "shower_dedx": shower_dedx, "shower_dedx_avg": shower_dedx_avg, "shower_dedx_cali" : shower_dedx_cali,
                      "track_dedx_hits": track_dedx_hits, "track_dedx": track_dedx, "track_dedx_avg": track_dedx_avg, "track_dedx_cali" : track_dedx_cali})

def OpticalInfo(row):
    flash_PE=0
    flash_time=0
    if np.any(row["flash_passed"]!=-1):
        flash_PE = row["flash_PE"][np.max(row["flash_passed"])]
        flash_time = row["flash_time"][np.max(row["flash_passed"])]
    return pd.Series({"flash_time": flash_time, "flash_PE": flash_PE})


def MatchedCleanup(row):    
    nu_shower_ids, nu_track_ids                = row["nu_shower_ids"],row["nu_track_ids"]
    nu_shower_daughters, nu_track_daughters    = row["nu_shower_daughters"],row["nu_track_daughters"]
    
    matched_tracks= row["matched_tracks"]
    matched_tracks_energy= row["matched_tracks_energy"]
    
    matched_showers= row["matched_showers"]
    matched_showers_energy= row["matched_showers_energy"]

    if len(np.hstack(nu_shower_daughters))>0:
        for daughters,pdg,energy in zip(nu_shower_daughters,matched_showers,matched_showers_energy):
            if len(daughters)>0:
                for daughter in daughters:
                    if daughter in nu_shower_ids:
                        index = np.where(nu_shower_ids==daughter)
                        matched_showers[index[0]]=pdg
                        matched_showers_energy[index[0]] = energy
                    if daughter in nu_track_ids:
                        index = np.where(nu_track_ids==daughter)
                        matched_tracks[index[0]]=pdg
                        matched_tracks_energy[index[0]] = energy
                        
    if len(nu_track_daughters)>0:
        if len(np.hstack(nu_track_daughters))>0:
            for daughters,pdg,energy in zip(nu_track_daughters,matched_tracks,matched_tracks_energy):
                if len(daughters)>0:
                    for daughter in daughters:
                        if daughter in nu_shower_ids:
                            index = np.where(nu_shower_ids==daughter)
                            matched_showers[index[0]]=pdg
                            matched_showers_energy[index[0]] = energy
                        if daughter in nu_track_ids:
                            index = np.where(nu_track_ids==daughter)
                            matched_tracks[index[0]]=pdg
                            matched_tracks_energy[index[0]] = energy

    
    return pd.Series({"matched_tracks": matched_tracks, "matched_tracks_energy": matched_tracks_energy,
                      "matched_showers": matched_showers, "matched_showers_energy": matched_showers_energy})
    


# ### Selection & Features dictionaries

# BNB $\nu$ + cosmics 
# 
# 
# selection: We want to keep only the passed events but no further cuts on truth information

# In[8]:

# Selection uses and ordered dict
Nu_select_true = OrderedDict([])
Nu_select_reco = OrderedDict([("Passed analyzer selection", lambda x: x["passed"]==1),
                              ("Reco vtx in fidVol", lambda row: CheckBorderTPC(*row[["vx","vy","vz"]],array=fid_arr))
                             ])

# Features uses a list of functions
Nu_feature_list =[DaughterInfo, ContainedRatio, ShowerChargeProfile, CalcRecoE, CalcDedx, OpticalInfo,MaxAngle,
                 CC_daughter_E, ShowerTrueEnd, TrueDaughterInfo, MatchedCleanup] # The last row used true info


# BNB $\nu_e$ intrinsic + cosmics 
# 
# 
# selection: We want to do an efficiency plot, therefore we will need to have the signal definition selection. Non passed events need also to be filtered on that requirement before they are saved. 
# 

# In[9]:

# Selection uses and ordered dict
Nue_select_true = OrderedDict([("Signal Definition 1e0p",true_thresholds_1e0p)])
Nue_select_reco = OrderedDict([("Passed analyzer selection", lambda x: x["passed"]==1)
                               #No fiducial reco cut since we need efficiency, so keep all events
                              ])

# Features uses a list of functions
Nue_feature_list =[DaughterInfo, ContainedRatio, ShowerChargeProfile, CalcRecoE, CalcDedx, OpticalInfo,MaxAngle,
                   CC_daughter_E, ShowerTrueEnd, TrueDaughterInfo, MatchedCleanup] # The last row used true info


# In-time cosmics 
# 
# 
# selection: Just passed events, no truth features

# In[10]:

# Selection uses and ordered dict
Intime_select_true = OrderedDict([])
Intime_select_reco = OrderedDict([("Passed analyzer selection", lambda x: x["passed"]==1),
                              ("Reco vtx in fidVol", lambda row: CheckBorderTPC(*row[["vx","vy","vz"]],array=fid_arr))
                             ])

# Features uses a list of functions
Intime_feature_list =[DaughterInfo, ContainedRatio, ShowerChargeProfile, CalcRecoE, CalcDedx, OpticalInfo, MatchedCleanup,MaxAngle]


# BNB ext or BNB data samples
# 
# 
# selection: we only want to keep passed events 
# 
# 
# We need to be carefull not adding features that depend on truth information.

# In[11]:

# Selection uses and ordered dict
Data_select_true = OrderedDict([])
Data_select_reco = OrderedDict([("Passed analyzer selection", lambda x: x["passed"]==1),
                                ("Reco vtx in fidVol", lambda row: CheckBorderTPC(*row[["vx","vy","vz"]],array=fid_arr))
                               ])
                                                                                              

# Features uses a list of functions
Data_feature_list =[DaughterInfo,ContainedRatio,ShowerChargeProfile,CalcRecoE,CalcDedx,OpticalInfo,MaxAngle]


# ### Main Function: loadData

# In[14]:

def loadData(filelist,                  # List of input files
             selection_true,            # Function dict that contains the true based selection, applied before saving the compact complete dataframw
             selection_reco,            # Ordered dictionary with bool function that act on rows
             featurelist,               # list of functions returning columns
             maxf=1,                    # Maximum number of files to loop over
             outputname="output"        # Name of the final picle file
            ):
    
    chunks = []                                 # list of small dataframes with all info
    chunks_nonpassed = []                       # list of small dataframes for failed event bookkeeping
    entries = 0                                 # entries before selection
    counter = np.zeros(len(selection_reco)+len(selection_true))      # counts number of events passing each stage
    total_pot = 0  # total POT of the sample
    chuncks_pot=0
    
    nfiles=len(filelist)
    if maxf<nfiles:
        nfiles=maxf
        
    print ("Start to load entries from",nfiles,"files.\n")
    start_time = time.time()
    
    progress=0
    for i_f,fname in enumerate(filelist[:nfiles]):
        
        #print(fname)
        # Store the POT of the sample
        try:
            df_pot = read_root(fname,"wouterNueCC/pot")
            temp_pot = df_pot['pot'].sum()
            chuncks_pot+=temp_pot
            total_pot+=temp_pot
            # Write this dataframe to a txtfile.
            df_pot[['run', 'subrun']].to_csv(r'/home/wouter/Documents/Jupyter/LEE_analyzer/Input/run_subrun_bnb.txt', header=None, index=None, sep=' ', mode='a')

            dftemp=read_root(fname,"wouterNueCC/pandoratree")
            #print(len(dftemp.index),'entries in',fname,'.')
            entries+=len(dftemp.index)
        except:
            print('Tree corrupt?', fname)
            continue
        
        # Truth based selection
        for index,(key, value) in enumerate(selection_true.items()):
            dftemp = dftemp[dftemp.apply(value,axis=1)]
            counter[index]+=len(dftemp.index)
            
        # Store some basic things about events that did not pass the selection! (but passed the truth selection)
        #dftemp_nonpassed = dftemp[dftemp["passed"]==0][["nu_pdg","nu_E","true_vx_sce","true_vy_sce","true_vz_sce",
        #                                                "category","bnbweight"]]
        # This compact dataframe should also have some optical information.
        #dftemp_nonpassed = pd.concat([dftemp_nonpassed, dftemp[dftemp["passed"]==0].apply(OpticalInfo,axis=1)], axis=1)
        #dftemp_nonpassed = pd.concat([dftemp_nonpassed, dftemp[dftemp["passed"]==0].apply(FlashPrecut,axis=1)], axis=1)
        #chunks_nonpassed.append(dftemp_nonpassed)
        
        # Reco based selection
        for index,(key, value) in enumerate(selection_reco.items()):
            dftemp = dftemp[dftemp.apply(value,axis=1)]
            counter[index+len(selection_true)]+=len(dftemp.index)
            
        #if no events passed, stop here
        if(len(dftemp.index)==0):
            continue
        
        # introduce the new flattened dataframe:
        df_new = dftemp[flat_columns]
        
        # Reduce the dataframe size of the vector columns
        for col in vec_columns:
            df_new[col]=dftemp[col].apply(safely_reduce_dtype)
        
        # add new features to it
        for value in featurelist:
            df_new = pd.concat([df_new, dftemp.apply(value,axis=1)], axis=1)
            
        chunks.append(df_new)
        
        if((i_f+1) % math.ceil(nfiles/10)==0 ):
            print ("Progress:",(progress+1)*10,"%.")
            split_output=False
            if(split_output):
                print("Concatenating output dataframes")
                #Reduce the dataframe size or the non vector columns
                print ("POT in this chunk:",str(chuncks_pot),"POT.")
                df = pd.concat(chunks,ignore_index=True,copy=False) 
                df,_ = reduce_mem_usage(df)
                df.to_pickle("../Input/"+outputname+"_"+str(progress)+".pckl")
                chunks=[]
                chuncks_pot=0
            progress+=1
        
       
    end_time = time.time()
    
    #Reduce the dataframe size or the non vector columns
    if len(chunks)>0:
        print("Concatenating last frame in case of failure, check for double")
        print ("POT in this chunk:",str(chuncks_pot),"POT.")
        df = pd.concat(chunks,ignore_index=True,copy=False) 
        df,_ = reduce_mem_usage(df)
        df.to_pickle("../Input/"+outputname+"_"+str(progress)+".pckl")
            
    print("\nSummary:")
    print (entries,"entries were loaded from",nfiles,"files, corresponding to",str(total_pot),"POT.")
    for key,counts in zip(list(selection_true.keys())+list(selection_reco.keys()),counter):
        print(counts,"Passed ",key," stage of selection.")
        
    print("\nLoading took ",sciNot(end_time-start_time)," seconds.")       
    
    #df_nonpassed = pd.concat(chunks_nonpassed,ignore_index=True,copy=False) 
    #df_nonpassed,_ = reduce_mem_usage(df_nonpassed)
    #df_nonpassed.to_pickle("../Input/"+outputname+"_nonpassed.pckl")
    
    print ("Final dataframe has",len(df.index),"entries.")
    
    end2_time = time.time()
    print("Pickling took ",sciNot(end2_time-end_time)," seconds.")


# ## Load dataframe and save to Pickle

# In[ ]:

loadData(filelist,          # List of input files
             Data_select_true,           # Function dict that contains the true based selection, applied before saving the compact complete dataframw
             Data_select_reco,           # Ordered dictionary with bool function that act on rows
             Data_feature_list,          # list of functions returning columns
             maxf=3200,                  # Maximum number of files to loop over
             outputname="bnbe"           # Name of the final picle file
            )
print("Done!")


# ### Done!

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



