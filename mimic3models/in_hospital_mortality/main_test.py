from __future__ import absolute_import
from __future__ import print_function
import os
import pandas as pd
import sys
import time
top_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# add the top-level directory of this project to sys.path so that we can import modules without error
sys.path.append(top_path)
import numpy as np
import argparse
import imp
import re
import gc
from os import listdir
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
import tensorflow as tf
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils
from keras.callbacks import ModelCheckpoint, CSVLogger
#python -m mimic3models.split_train_val data/in-hospital-mortality
#   python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --save_every 10000 --output_dir mimic3models/in_hospital_mortality

#                   -------- argparse --------
parser = argparse.ArgumentParser()
# WTF !
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)

parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/')))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args()
physical_devices = tf.config.list_physical_devices('GPU')
# Don't pre-allocate memory; allocate as-needed
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#                   --------------------------

#                   ------ not yet realize ----
if args.small_part:   #TODO:
    args.save_every = 2**30
def get_loc(fn):
    a=re.match(r'.+(0.001).+seed_([0-9]).*',fn)
    b=re.match(r'.+(0.01).+seed_([0-9]).*',fn)
    c=re.match(r'.+(0.05).+seed_([0-9]).*',fn)
    d=re.match(r'.+(1.0).+seed_([0-9]).*',fn)
    if a:
        return int(a[2]),0
    elif b:
        return int(b[2]),1
    elif c:
        return int(c[2]),2
    elif d:
        return int(d[2]),2
    else:
        return None

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train') #TODO:
#                   ---------------------------
run_time=time.strftime("-%Y-%m-%d_%H_%M_%S")
# percentage = float(args.percentage)
# sampleTimes = int(args.sampletimes)
accs=pd.DataFrame(index=list(range(10)),columns=[0.001,0.01,0.05,1,0])
roc_tests = pd.DataFrame(index=list(range(10)),columns=[0.001,0.01,0.05,1,0])
prc_tests = pd.DataFrame(index=list(range(10)),columns=[0.001,0.01,0.05,1,0])
pses = pd.DataFrame(index=list(range(10)),columns=[0.001,0.01,0.05,1,0])
train_list =pd.read_csv(os.path.join(top_path,'ref','train_listfile.csv'))
for fn in listdir('/home/shuying/cpcSurvival/scripts/nohup/2080/mimic3models/in_hospital_mortality/keras_states'):
    if '0.001' not in fn and '0.05' not in fn and '0.01' not in fn: continue
    i,j=get_loc(fn)
    print(fn)
    temp_list_file = train_list[train_list['0.001'] == 1]
    temp_list_file=temp_list_file.loc[:,['stay','y_true']]
    temp_list_file.to_csv(os.path.join(args.data, 'temp.csv'),index=False)
# Build readers, discretizers, normalizers
    train_reader = InHospitalMortalityReader(dataset_dir=args.data,
                                             listfile=os.path.join(args.data, 'temp.csv'),
                                             period_length=48.0)

    # * mined that dataset_dir of it is 'train'

    # default impute strategy is 'previous' here
    # default timestep is 1.0
    discretizer = Discretizer(timestep=float(args.timestep),
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')

    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]  # only categorical data has '->'

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)  # ?Need to run "create_normalizer_state.py" first

    args_dict = dict(args._get_kwargs())
    args_dict['header'] = discretizer_header
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl

    # Build the model
    model_module = imp.load_source(os.path.basename(args.network), args.network)
    model = model_module.Network(**args_dict)
    suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                       ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                       ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                       args.timestep,
                                       ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
    model.final_name = args.prefix + model.say_name() + suffix


    # Compile the model
    # print("==> compiling the model")

    optimizer_config = {'class_name': args.optimizer,
                        'config': {'learning_rate': args.lr,
                                   'beta_1': args.beta_1}}
    # NOTE: one can use binary_crossentropy even for (B, T, C) shape.
    #       It will calculate binary_crossentropies for each class
    #       and then take the mean over axis=-1. Tre results is (B, T).
    if target_repl:
        loss = ['binary_crossentropy'] * 2
        loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
    else:
        loss = 'binary_crossentropy'
        loss_weights = None

    model.compile(optimizer=optimizer_config,
                  loss=loss,
                  loss_weights=loss_weights)

    # Load model weights
    n_trained_chunks = 0
    model.load_weights(os.path.join('/home/shuying/cpcSurvival/scripts/nohup/2080/mimic3models/in_hospital_mortality/keras_states',fn))


    # Read data
    train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part,percentage=1)

    if target_repl:
        T = train_raw[0][0].shape[0]

        def extend_labels(data):
            data = list(data)
            labels = np.array(data[1])  # (B,)
            data[1] = [labels, None]
            data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
            data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
            return data

        train_raw = extend_labels(train_raw)


    # ensure that the code uses test_reader
    del train_reader
    del train_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    results=metrics.print_metrics_binary(labels, predictions)
    auroc=results['auroc']
    auprc=results['auprc']
    minpse=results['minpse']
    acc=results['acc']
    roc_tests.iloc[i,j]=auroc
    prc_tests.iloc[i,j]=auprc
    pses.iloc[i,j]=minpse
    accs.iloc[i,j]=acc
    path = os.path.join(args.output_dir, "test_", os.path.basename(fn)) + ".csv"
    utils.save_results(names, predictions, labels, path)
    del temp_list_file,model
    gc.collect()

print('auroc-------')
print(roc_tests)
print('auprc-------')
print(prc_tests)
print('pses-------')
print(pses)
print('accs=====')
print(acc)
roc_tests.to_csv('/home/shuying/cpcSurvival/scripts/nohup/2080/mimic3models/in_hospital_mortality/keras_logs/auroc.csv')
prc_tests.to_csv('/home/shuying/cpcSurvival/scripts/nohup/2080/mimic3models/in_hospital_mortality/keras_logs/auprc.csv')
pses.to_csv('/home/shuying/cpcSurvival/scripts/nohup/2080/mimic3models/in_hospital_mortality/keras_logs/pse.csv')
accs.to_csv('/home/shuying/cpcSurvival/scripts/nohup/2080/mimic3models/in_hospital_mortality/keras_logs/acc.csv')
