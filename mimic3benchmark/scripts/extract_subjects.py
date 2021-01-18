from __future__ import absolute_import
from __future__ import print_function
"""
python -m mimic3benchmark.scripts.extract_subjects /home/shuying/mimic3/files/mimiciii/1.4 data/root/

"""
import argparse
import yaml
import sys
sys.path.append('/home/shuying/survival/benchmark/mimic3-benchmarks/')
from mimic3benchmark.mimic3csv import *
from mimic3benchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from mimic3benchmark.util import dataframe_from_csv

# ------ arg parser -----

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                    default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'])
parser.add_argument('--phenotype_definitions', '-p', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccs_2015_definitions.yaml'),
                    help='YAML file with phenotype definitions.')
parser.add_argument('--itemids_file', '-i', type=str, help='CSV containing list of ITEMIDs to keep.')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output')
parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suspend printing of details')
parser.set_defaults(verbose=True)
parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1,000 subjects, 1,000,000 events.')
args, _ = parser.parse_known_args()


# -----------------------

# create output dir
try:
    os.makedirs(args.output_path)
except:
    pass

# ----- load in data ----

# function defined in mimic3csv.py
patients = read_patients_table(args.mimic3_path)
admits = read_admissions_table(args.mimic3_path)
stays = read_icustays_table(args.mimic3_path)

# print initial state : the nonredundant count 
if args.verbose:
    print('START:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))
    
# -----------------------


# -----remove transfered patients-----

stays = remove_icustays_with_transfers(stays)

# print current state : remaining number of each metrics
if args.verbose:
    print('REMOVE ICU TRANSFERS:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

# concatenating info from `admits` and `patients` that has no transfer record
stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)

# ------------------------------------



# ==| filter by the times entering icu |==
stays = filter_admissions_on_nb_icustays(stays)
if args.verbose:
    print('REMOVE MULTIPLE STAYS PER ADMIT:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

    
    
# --------- adding AGE , MORALITY and screening of age ---------

stays = add_age_to_icustays(stays)                               # add age info
stays = add_inunit_mortality_to_icustays(stays)                  # die in ICU ? 0: alive , 1: dead
stays = add_inhospital_mortality_to_icustays(stays)              # die in hospital ? 0 : alive , 1: dead
stays = filter_icustays_on_age(stays)                            # screening patients older than 18

# reporting current state 

if args.verbose:
    print('REMOVE PATIENTS AGE < 18:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

# ------------------------------------- -------------------------





# -----------------------\               /--------------------------
# ======================== SAVE to new csv =========================
# -----------------------/               \--------------------------

stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)

# ==================================================================



# read diagnose data passed from `D_ICD_DIAGNOSES.csv`  and  `DIAGNOSES_ICD.csv` 
diagnoses = read_icd_diagnoses_table(args.mimic3_path)
# screen patients
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)

diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)

# summarize the number of icd codes (disease) and count the corresponding patients
count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))

# A dataframe similar to diagnoses but with two more columns 
phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.load(open(args.phenotype_definitions, 'r')))
# generate a matrix with ICUSTAY_ID to its disease code, reindexed , and save to new csv 
make_phenotype_label_matrix(phenotypes, stays).to_csv(os.path.join(args.output_path, 'phenotype_labels.csv'),
                                                      index=False, quoting=csv.QUOTE_NONNUMERIC)

# TEST MODE: process only 1,000 subjects, 1,000,000 events.
if args.test:
    pat_idx = np.random.choice(patients.shape[0], size=1000)
    patients = patients.iloc[pat_idx]
    stays = stays.merge(patients[['SUBJECT_ID']], left_on='SUBJECT_ID', right_on='SUBJECT_ID')
    args.event_tables = [args.event_tables[0]]
    print('Using only', stays.shape[0], 'stays and only', args.event_tables[0], 'table')

subjects = stays.SUBJECT_ID.unique()
# generate info for each patients
break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
break_up_diagnoses_by_subject(phenotypes, args.output_path, subjects=subjects)

items_to_keep = set(
    [int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['ITEMID'].unique()]) if args.itemids_file else None

# default setting contain 3 events
for table in args.event_tables:
    read_events_table_and_break_up_by_subject(args.mimic3_path, table, args.output_path, items_to_keep=items_to_keep,
                                              subjects_to_keep=subjects)
