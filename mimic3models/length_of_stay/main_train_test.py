from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import argparse
import os
import imp
import re
top_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import sys
sys.path.append(top_path)
from mimic3models.length_of_stay import utils
from mimic3benchmark.readers import Structure_map_Dataset
import gc
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger








parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--partition', type=str, default='custom',
                    help="log, custom, none")
parser.add_argument('--data', type=str, help='Path to the data of length-of-stay task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/length-of-stay/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--seed',default=0,type=int)

args = parser.parse_args()
prefix='lstm'+'_seed_%d'%args.seed
suffix='.csv'
auroc_path=prefix+'_auroc_'+suffix
acc_path=prefix+'_acc_'+suffix
kappa_path=prefix+'_kappa_'+suffix
print(args)
num_per_class=[10,50,None]
total=[None,None,None]
names=[100,500,'All']
auroc_tests = pd.DataFrame(index=[args.seed],columns=names)
acc_tests = pd.DataFrame(index=[args.seed],columns=names)
kappa_tests = pd.DataFrame(index=[args.seed],columns=names)
if args.small_part:
    args.save_every = 2**30
seed=args.seed
physical_devices = tf.config.list_physical_devices('GPU')
# Don't pre-allocate memory; allocate as-needed
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Build readers, discretizers, normalizers
for i in range(3):
    print(i)
    print('There are %s samples in the training set'%names[i])
    train_reader = Structure_map_Dataset(data_dir=os.path.join(args.data, 'train'),
                                      list_file=os.path.join(args.data, 'train_listfile.csv'),
                                      num_per_class=num_per_class[i],
                                      total=total[i],
                                      random_seed=seed)
    train_x=train_reader.get_x()
    train_y=train_reader.get_true()
    print("get training data done")
    val_reader = Structure_map_Dataset(data_dir=os.path.join(args.data, 'train'),
                                    list_file=os.path.join(args.data, 'val_listfile.csv'),
                                    num_per_class=10,
                                    total=100,
                                    random_seed=seed)
    val_x=val_reader.get_x()
    val_y=val_reader.get_true()
    print("get validation data done")

    discretizer = Discretizer(timestep=args.timestep,
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')

    # discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    # cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
    #
    # normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    # normalizer_state = args.normalizer_state
    # if normalizer_state is None:
    #     normalizer_state = 'los_ts{}.input_str:previous.start_time:zero.n5e4.normalizer'.format(args.timestep)
    #     normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    # normalizer.load_params(normalizer_state)
    args_dict = dict(args._get_kwargs())
    # args_dict['header'] = discretizer_header
    args_dict['task'] = 'los'
    args_dict['num_classes'] = (1 if args.partition == 'none' else 10)


    # Build the model
    print("==> using model {}".format(args.network))
    model_module = imp.load_source(os.path.basename(args.network), args.network)
    model = model_module.Network(**args_dict)
    suffix = '%s_seed_%d'%(names[i],seed)
    model.final_name = 'lstm_'+suffix
    print("==> model.final_name:", model.final_name)


    # Compile the model
    print("==> compiling the model")
    optimizer_config = {'class_name': args.optimizer,
                        'config': {'lr': args.lr,
                                   'beta_1': args.beta_1}}

    loss_function = 'categorical_crossentropy'
    # NOTE: categorical_crossentropy needs one-hot vectors
    #       that's why we use sparse_categorical_crossentropy
    # NOTE: it is ok to use keras.losses even for (B, T, D) shapes

    model.compile(optimizer=optimizer_config,
                  loss=loss_function)

    model.summary()


    # Load model weights
    n_trained_chunks = 0
    #

    #==========TRAINING ================
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.chunk{epoch}.test{val_loss}.state')

    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


    print("==> training")
    model.fit(x=train_x,
              y=train_y,
              validation_data=(val_x,val_y),
              batch_size=args.batch_size,epochs=args.epochs,verbose=1
       )



    # ==================TEST======================================
        # ensure that the code uses test_reader

    labels = []
    predictions = []

    test_reader = Structure_map_Dataset(data_dir=os.path.join(args.data, 'test'),
                                     list_file=os.path.join(args.data, 'test_listfile.csv'))
    test_x=test_reader.get_x()
    print("get test data done")
    preds=model.predict(test_x,batch_size=100)
    print('get test predictions done')
    labels=test_reader.get_true()
    auroc= metrics.metrics.roc_auc_score(labels,preds,multi_class='ovo')
    acc=metrics.metrics.accuracy_score(np.argmax(preds,1),np.argmax(labels,1))
    kappa=metrics.metrics.cohen_kappa_score(np.argmax(preds,1),np.argmax(labels,1))



    auroc_tests.iloc[0, i] = auroc
    acc_tests.iloc[0, i] = acc
    kappa_tests.iloc[0, i] = kappa
    gc.collect()
    del train_reader,preds,labels,val_reader,test_reader,model



    auroc_tests.to_csv(os.path.join(top_path, 'logs', 'los', 'LSTM',auroc_path))
    acc_tests.to_csv(os.path.join(top_path, 'logs', 'los', 'LSTM',acc_path))
    kappa_tests.to_csv(os.path.join(top_path, 'logs', 'los', 'LSTM',kappa_path))
    print('AUROC for test updated in %s'%auroc)
    print('ACC for test updated in %s'%acc)
    print('KAPPA for test updated in %s'%kappa)