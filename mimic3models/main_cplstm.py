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
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
import tensorflow as tf
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models.keras_models.lstm import CPLSTM
from mimic3models import common_utils
from keras.callbacks import ModelCheckpoint, CSVLogger
import random
import time

#python -m mimic3models.split_train_val data/in-hospital-mortality
#   python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --save_every 10000 --output_dir mimic3models/in_hospital_mortality

#                   -------- argparse --------
run_time=time.strftime("-%Y-%m-%d_%H_%M_%S")
parser = argparse.ArgumentParser()
# WTF !
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)

parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/')))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--predstep',type=int,default=5)
args = parser.parse_args()
physical_devices = tf.config.list_physical_devices('GPU')
# Don't pre-allocate memory; allocate as-needed
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#                   --------------------------

#                   ------ not yet realize ----
if args.small_part:   #TODO:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train') #TODO:
#                   ---------------------------
accs=[]
# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=args.data,
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

# * mined that dataset_dir of it is 'train'
val_reader = InHospitalMortalityReader(dataset_dir=args.data,
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

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
# args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'

# Build the model
print("==> using model {}".format(args.network))
# model_module = imp.load_source(os.path.basename(args.network), args.network)
model = CPLSTM(**args_dict)
suffix ="run_time_%s_timestep_%s"%(run_time,args_dict['predstep'])

model.final_name = args.prefix + '_CPLSTM_'+ suffix
print("==> model.final_name:", model.final_name)


# Compile the model
# print("==> compiling the model")

optimizer_config = {'class_name': args.optimizer,
                    'config': {'learning_rate': args.lr,
                               'beta_1': args.beta_1}}
# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).


model.compile(optimizer="Adam", loss="mse"
            )
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


# Read data
train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part,percentage=1)
val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)

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
    val_raw = extend_labels(val_raw)

if args.mode == 'train':

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name +'_'+run_time+ '.epoch{epoch}.test{val_loss}.state')


    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '_'+run_time+'.csv'),
                           append=True, separator=';')
    model.fit(x=train_raw[0],
              y=np.zeros(len(train_raw[0])),
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[ saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)
    gc.collect()

