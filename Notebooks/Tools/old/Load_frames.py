#!/usr/bin/python
# -*- coding: utf-8 -*-

## LEE Analyzer notebook

# This notebook takes in the output directory of the jobs and 
# convert it into a more flat pandas dataframe.
# Data from the root files will be partially processed 
# to fields that are convenient to plot.
# The resulting dataframe will be pickled.

### Imports & Constants
# To make it compatible with the grid

from __future__ import division
from __future__ import print_function

import math
import time
import os
import sys
import glob
import numpy as np
import pandas as pd
from root_pandas import read_root
from helpfunction import sciNot

pd.options.display.max_columns = 999
min_root_size = 20000  # Skip root files smaller than x bytes

vtx_activity_cut = 5  # how many objects start withing 5 cm of the vertex?
z_dead_start = 675
z_dead_end = z_dead_start + 100


# Flat columns we want to copy from the original dataframe:

flat_columns_truth = [
    'nu_pdg',
    'nu_E',
    'true_vx_sce',
    'true_vy_sce',
    'true_vz_sce',
    'distance',
    'ccnc',
    'qsqr',
    'theta',
    'true_1eX_signal',
    'true_nu_fiducial',
    'lepton_E',
    'lepton_theta',
    'true_vx',
    'true_vy',
    'true_vz',
    ]

flat_columns_reco = [
    'event',
    'subrun',
    'run',
    'category',
    'vx',
    'vy',
    'vz',
    'bnbweight',
    'passed',
    'candidate_pdg',
    'numu_cuts',
    'track_bdt_precut',
    'n_showers',
    'n_tracks',
    'flash_time_max',
    'flash_PE_max',
    'chargecenter_x',
    'chargecenter_y',
    'chargecenter_z',
    'total_spacepoint_containment',
    'fiducial',
    ]

vec_columns_shower = [
    'shower_open_angle',
    'shower_length',
    'shower_start_x',
    'shower_start_y',
    'shower_start_z',
    'shower_dir_x',
    'shower_dir_y',
    'shower_dir_z',
    'shower_pca',
    'shower_maxangle',
    'shower_vtxdistance',
    'shower_daughter',
    'shower_is_daughter',
    'shower_fidvol_ratio',
    'shower_spacepoint_dqdx_ratio',
    'shower_dedx_hits_w',
    'shower_dedx_w',
    'shower_dedx_best_w',
    'shower_energy_w',
    'shower_hitsratio_w',
    'shower_hits_w',
    'shower_theta',
    'shower_phi',
    'shower_energy_product',
    ]

vec_columns_track = [  
    'track_start_x',
    'track_start_y',
    'track_start_z',
    'track_end_x',
    'track_end_y',
    'track_end_z',
    'track_dir_x',
    'track_dir_y',
    'track_dir_z',
    'track_pca',
    'predict_em',
    'predict_mu',
    'predict_cos',
    'predict_pi',
    'predict_p',
    'track_res_mean',
    'track_res_std',
    'track_maxangle',
    'track_vtxdistance',
    'track_daughter',
    'track_is_daughter',
    'track_spacepoint_dqdx_ratio',
    'track_containment',
    'track_dedx_hits_w',
    'track_dedx_w',
    'track_dedx_best_w',
    'track_energy_w',
    'track_hitsratio_w',
    'track_hits_w',
    'track_theta',
    'track_len',
    'track_phi',
    ]

vec_columns_truth = [
    'true_shower_pdg',
    'true_shower_x_sce',
    'true_shower_y_sce',
    'true_shower_z_sce',
    'true_shower_depE',
    'shower_cle',
    'matched_showers',
    'matched_showers_energy',
    'track_cle',
    'matched_tracks',
    'matched_tracks_energy',
    'nu_daughters_pdg',
    'nu_daughters_E',
    ]

# Columns to use for main frame

columns_data = flat_columns_reco + vec_columns_shower + vec_columns_track
columns_mc = columns_data + flat_columns_truth + vec_columns_truth

# Columns to use for track/shower frame

columns_shower_mc = vec_columns_shower + ['shower_cle', 'matched_showers', 'matched_showers_energy']
columns_track_mc = vec_columns_track + ['track_cle', 'matched_tracks', 'matched_tracks_energy']

columns_flat = [
    'bnbweight',
    'noexpand:1<(n_showers+n_tracks)',
    'fiducial',
    'track_bdt_precut',
    'n_showers',
    'n_tracks',
    'event',
    'subrun',
    'run',
    'candidate_pdg',
    'numu_cuts',
    'category',
    ]

### Select Input Files


inputlist = []
inputlist += glob.glob('/uboone/data/users/wvdp/v06_26_01_12/April/bnb_nue_cosmic_0s0t/*.root')
# inputlist += glob.glob("/uboone/data/users/wvdp/v06_26_01_12/April/data_bnbext_b_0s0t/*.root")

filelist = []
for fname in inputlist:
    if os.stat(fname).st_size < min_root_size:
        print('File', fname, 'was corrupt. Size:',
              os.stat(fname).st_size / 1000,
              'kb, skipping to next file.')
    else:
        filelist.append(fname)
print(len(filelist), 'valid ROOT files collected.')


### Functions

#### Main Function: loadData
# list,   Input files.
# bool,   Apply the true signal and save nonpassed events.
# bool,   If this is data, save less stuff
# bool,   Split output in 10 dataframes.
# int,    Maximum number of files to loop over.
# string, Name of the final picle file.
def loadData(  
    filelist,
    signal_sample,
    data,
    split_output,
    maxf=1,
    outputname='output',
    ):

                                 

    chunks = []  # list of small dataframes with all info
    chunks_tr = []
    chunks_sh = []
    chunks_nonpassed = []  # list of small dataframes for failed event bookkeeping

    entries = 0  # entries before selection
    entries_signal = 0
    flash_passed = 0
    passed = 0
    bdt_precut_passed = 0
    fidvol = 0
    cat2 = 0
    non_passed = 0

    total_pot = 0  # total POT of the sample
    chuncks_pot = 0

    columns_load = columns_data
    columns_track = vec_columns_track
    columns_shower = vec_columns_shower
    if not data:
        global columns_flat
        columns_track = columns_track_mc
        columns_shower = columns_shower_mc
        columns_load = columns_mc
        columns_flat += ['true_1eX_signal', 'lepton_theta', 'lepton_E', 'true_vz']

    nfiles = len(filelist)
    if maxf < nfiles:
        nfiles = maxf

    print('Start to load entries from', nfiles, 'files.\n')
    start_time = time.time()

    progress = 0
    for (i_f, fname) in enumerate(filelist[:nfiles]):
        try:

            # Store the POT of the sample

            df_pot = read_root(fname, 'wouterNueCC/pot')
            temp_pot = df_pot['pot'].sum()
            chuncks_pot += temp_pot
            total_pot += temp_pot

            # Write this dataframe to a txtfile.

            df_pot[['run', 'subrun']].to_csv('../Input/' + outputname
                    + '/run_subrun.txt', header=None, index=None,
                    sep=' ', mode='a')
            dftemp = read_root(fname, 'wouterNueCC/pandoratree',
                               columns=columns_load)
            entries += len(dftemp.index)

            # Track/Shower frames

            df_tr = read_root(fname, 'wouterNueCC/pandoratree',
                              columns=columns_track + columns_flat,
                              flatten=columns_track)
                              
            df_sh = read_root(fname, 'wouterNueCC/pandoratree',
                              columns=columns_shower + columns_flat,
                              flatten=columns_shower)
        except BaseException, e:

            print('Tree corrupt?', fname, '\n', str(e))
            continue

        str_eval_unresponsive_z = 'unresponsive_z = ~( @z_dead_start < true_vz < @z_dead_end)'

        if signal_sample:
            dftemp = dftemp.query('true_1eX_signal==1')

            dftemp.eval(str_eval_unresponsive_z, inplace=True)
            dftemp = dftemp.query('unresponsive_z==1')

            entries_signal += dftemp['bnbweight'].sum()

            # Store some basic things about events that did not pass the selection! 
            # (but passed the truth selection)

            str_query = 'n_showers<1 or candidate_pdg!=12 or fiducial==0 or unresponsive_z==0'

            dftemp_nonpassed = dftemp.query(str_query, inplace=False)
                                     [flat_columns_reco + flat_columns_truth]

            chunks_nonpassed.append(dftemp_nonpassed)
            non_passed += dftemp_nonpassed['bnbweight'].sum()

            df_tr.eval(str_eval_unresponsive_z, inplace=True)
            df_sh.eval(str_eval_unresponsive_z, inplace=True)

            str_query = 'true_1eX_signal==1 and unresponsive_z==1'
            df_tr.query(str_query, inplace=True)
            df_sh.query(str_query, inplace=True)

        if data or outputname == 'intime':
            dftemp['bnbweight'] = 1

        flash_passed += dftemp.query('flash_time_max>0')['bnbweight'
                ].sum()

        dftemp.query('n_showers>0 & passed==1', inplace=True)
        passed += dftemp['bnbweight'].sum()

        dftemp.query('candidate_pdg==12', inplace=True)
        bdt_precut_passed += dftemp['bnbweight'].sum()

        dftemp.query('fiducial==1', inplace=True)
        fidvol += dftemp['bnbweight'].sum()

        cat2 += dftemp.query('category==2')['bnbweight'].sum()

        str_query = 'candidate_pdg==12 & fiducial==1 & n_showers>0'
        df_tr.query(str_query, inplace=True)
        df_tr.rename(columns={'1<(n_showers+n_tracks)': 'vtx_activity'}, inplace=True)
        df_sh.query(str_query, inplace=True)
        df_sh.rename(columns={'1<(n_showers+n_tracks)': 'vtx_activity'}, inplace=True)

        # if no events passed, stop here

        if len(dftemp.index) == 0:
            continue

        # Add a feature:

        dftemp['vtx_activity_nr'] = dftemp.apply(lambda x: \
                sum(x['shower_vtxdistance'] < vtx_activity_cut) \
                + sum(x['track_vtxdistance'] < vtx_activity_cut),
                axis=1)

        chunks.append(dftemp)
        chunks_tr.append(df_tr)
        chunks_sh.append(df_sh)

        if (i_f + 1) % math.ceil(nfiles / 10) == 0:
            print('Progress:', (progress + 1) * 10, '%.')
            if split_output:
                print('Concatenating output dataframes')

                # Reduce the dataframe size or the non vector columns

                print('POT in this chunk:', str(chuncks_pot), 'POT.')
                df = pd.concat(chunks, ignore_index=True, copy=False)
                df.to_pickle('../Input/' + outputname + '/'
                             + outputname + '_' + str(progress)
                             + '.pckl')
                chunks = []
                chuncks_pot = 0

                df = pd.concat(chunks_sh, ignore_index=True, copy=False)
                df.to_pickle('../Input/' + outputname + '/'
                             + outputname + '_shower_' + str(progress)
                             + '.pckl')
                chunks_sh = []
                df = pd.concat(chunks_tr, ignore_index=True, copy=False)
                df.to_pickle('../Input/' + outputname + '/'
                             + outputname + '_track_' + str(progress)
                             + '.pckl')
                chunks_tr = []
            progress += 1

    end_time = time.time()

    if len(chunks) > 0:
        print('Concatenating last frame in case of failure, check for double'
              )
        print('POT in this chunk:', str(chuncks_pot), 'POT.')
        df = pd.concat(chunks, ignore_index=True, copy=False)
        df.to_pickle('../Input/' + outputname + '/' + outputname + '_' + str(progress) + '.pckl')
        chunks = []
        df = pd.concat(chunks_sh, ignore_index=True, copy=False)
        df.to_pickle('../Input/' + outputname + '/' + outputname + '_shower_' + str(progress) + '.pckl')
        chunks_sh = []
        df = pd.concat(chunks_tr, ignore_index=True, copy=False)
        df.to_pickle('../Input/' + outputname + '/' + outputname + '_track_' + str(progress) + '.pckl')
        chunks_tr = []

    print('\nSummary:')
    print(
        entries,
        'entries were loaded from',
        nfiles,
        'files, corresponding to',
        str(total_pot),
        'POT.',
        )
    if signal_sample:
        print(int(entries_signal), 'events are 1eX signal in fidvol.')
        if non_passed + fidvol != entries_signal:
            print(
                'ERROR: the passing (',
                fidvol,
                ') and non-passing (',
                non_passed,
                ') events did not sum up correctly to',
                entries_signal,
                '!',
                )

    print(int(flash_passed), ' events pass the optical precuts.')
    print(int(passed), 'events pass the selection (if topo).')
    print(int(bdt_precut_passed), 'events pass the track bdt precut.')
    print(int(fidvol), 'events are in the fiducial volume.')
    print(int(cat2), 'events are category electron neutrino.')

    print('\nLoading took ', sciNot(end_time - start_time), ' seconds.')

    if signal_sample:
        df_nonpassed = pd.concat(chunks_nonpassed, ignore_index=True, copy=False)
        df_nonpassed.to_pickle('../Input/' + outputname + '/' + outputname + '_nonpassed.pckl')

    end2_time = time.time()
    print('Pickling took ', sciNot(end2_time - end_time), ' seconds.')
    print('Done!')


### Load dataframe and save to Pickle
# list,   Input files.
# bool,   Apply the true signal and save nonpassed events.
# bool,   If this is data, save less stuff
# bool,   Split output in 10 dataframes.
# int,    Maximum number of files to loop over.
# string, Name of the final picle file.
           
loadData( 
    filelist,
    signal_sample=True,
    data=False,
    split_output=False,
    maxf=5000,
    outputname='nue',
    )

