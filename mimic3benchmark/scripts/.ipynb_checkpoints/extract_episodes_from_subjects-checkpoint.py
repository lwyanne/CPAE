from __future__ import absolute_import
from __future__ import print_function


import argparse
import os
import sys
from numba import jit
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mimic3benchmark.subject import read_stays, read_diagnoses, read_events, get_events_for_stay,\
    add_hours_elpased_to_events
from mimic3benchmark.subject import convert_events_to_timeseries, get_first_valid_from_timeseries
from mimic3benchmark.preprocessing import read_itemid_to_variable_map, map_itemids_to_variables, clean_events
from mimic3benchmark.preprocessing import assemble_episodic_data,given_ITEMID

#                                  -----| argparser |-----
parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/itemid_to_variable_map.csv'),
                    help='CSV containing ITEMID-to-VARIABLE map.')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for VARIABLEs.')
parser.add_argument('--variables_to_keep', type=str,
                    default=None,
                    help='A txt file containing ITEMID to stay in episode Matrix') 
                    # os.path.join(os.path.dirname(__file__), '../../ref/variableList.txt')
parser.add_argument('--STATUS', type=str,
                    default='ready',
                    help='the STATUS of variables to keep,can be multiple string seperate by "," or Even empty string')

args, _ = parser.parse_known_args()
#                                  -----==============-----

if args.variables_to_keep is not None:
    # new function
    var_map,variables = given_ITEMID(args.variable_map_file,args.variables_to_keep)
else:
    # a strong selection of `ITEM_ID` here, only 114 was read in 
    var_map = read_itemid_to_variable_map(args.variable_map_file,STATUS=args.STATUS)
    variables = var_map.VARIABLE.unique()


for subject_dir in tqdm(os.listdir(args.subjects_root_path), desc='Iterating over subjects'):
    dn = os.path.join(args.subjects_root_path, subject_dir)
    
    #                     -------    read file by subject id    -------
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):            
            raise Exception  # check dir
    except: 
        continue

    try:
        # reading tables of this subject
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events = read_events(os.path.join(args.subjects_root_path, subject_dir))
    except:
        sys.stderr.write('Error reading from disk for subject: {}\n'.format(subject_id))
        continue
    #                    ------------------------------------------------

    # pivot table containing 128 diagnosis label and stays info, index by `Icustay`
    episodic_data = assemble_episodic_data(stays, diagnoses)

    #  ------------｜ screen itemid  |------------
    # cleaning and converting to time series
    events = map_itemids_to_variables(events, var_map)
    events = clean_events(events)
    #            ---------------------

    if events.shape[0] == 0:
        # no valid events for this subject
        continue

    # ------------------|   the 17 itemid(variables) is fixed here   |----------
    #  
    timeseries = convert_events_to_timeseries(events, variables=variables)
    #                     --------------------------------------------


    
    # extracting separate episodes
    for i in range(stays.shape[0]):
        stay_id = stays.ICUSTAY_ID.iloc[i]
        intime = stays.INTIME.iloc[i]
        outtime = stays.OUTTIME.iloc[i]

        # extract a episode within given time
        episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
        if episode.shape[0] == 0:
            # no data for this episode
            continue
        
        # convert CHARTTIME to HOURS 
        episode = add_hours_elpased_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)
        if stay_id in episodic_data.index:
            episodic_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode, 'Weight')
            episodic_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode, 'Height')
        # episode.csv is a ICUstay info with diagnose info
        episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir,
                                                                              'episode{}.csv'.format(i+1)),
                                                                 index_label='Icustay')
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i+1)),
                       index_label='Hours')
        # episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_mytimeseries.csv'.format(i+1)),
                    #    index_label='Hours')