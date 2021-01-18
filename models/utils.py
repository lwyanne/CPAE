""" This module contains code to handle data """
import os, sys
import re
import numpy as np
import inspect
import pandas as pd
import platform
import pickle
import json
import datetime as dt
from tqdm import tqdm
import configparser
import torch
import logging
import argparse
from timeit import default_timer as timer
from torch.utils.data import random_split
import scipy
import time
import pathlib
from os import listdir
import json
import random

# global variable

global device  # automatically determine to use cpu or gpu(cuda)
global top_path  # the path of the top_level directory
global data_dir, script_dir, logging_dir
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# add the top-level directory of this project to sys.path so that we can import modules without error
top_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(top_path)


def get_machine_dir(file=os.path.join(top_path, 'machineConfig.json')):
    with open(file, 'r') as f:
        mydict = json.load(f)
        return mydict['script_dir'], mydict['data_dir'], mydict['logging_dir']


def get_lr_ifbn(args,cpc_args):
    try:lrs = args['lrs']
    except KeyError: lrs=args['lr']
    if args['percentage']==0.001 and 'LSTM' in cpc_args['model_type']:
        lrs=1e-2
    ifbn=args.get('ifbn')
    if args['freeze'] and args['percentage']==1:
        lrs=1e-2
    print('learning rate : %s'%str(lrs))
    if ifbn: print('Batch Normalization : True')
    return lrs,ifbn

def get_bs_imp(args,len_train_set):
    bs = args['bs']
    if len_train_set < bs:
        bs = 5
    print('Batch size : %s' %bs)
    return bs

def load_only_encode(model,states):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in states.items() if 'lstm1' in k}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def load_model_imp(cpc,states,args):
    freeze_partial = None
    if args['pre_train']:
        if args.get('loadreg') or args.get('loadreg') is None:
            cpc.load_state_dict(states)
            print('\n-----------\nload parameters of pretrained model.lstm1 & lstm2....\n')
        else:
            load_only_encode(cpc, states)
            print('\n-----------\nOnly load parameters of pretrained model.lstm1.....\n')  # load
            if args['freeze']:
                freeze_partial = 'encode'
                print('model.lstm1 is freeze')
    return freeze_partial



def discriminate_leaner(myLearner,l,model_type):
    if 'LSTM' in model_type:
        l = [myLearner.model.CPmodel.lstm1, myLearner.model.CPmodel.lstm2, myLearner.model.MLP]
        myLearner.split(split_on=l)

def define_save_path_imp(args,lrs):
    bestLearner=os.path.join(data_dir,'models','imp', '%s_%s_seed_%s_modelseed_%s_update_%s_freeze_%s_lr_%s-best' % (
    args['setting_name'], args['percentage'], args['seed'], args['model_seed'], args['update_mseed'], args['freeze'], lrs))

    tempLearner=os.path.join(data_dir,'models','imp', '%s_%s_seed_%s_modelseed_%s_update_%s_freeze_%s_lr_%s-temp' % (
    args['setting_name'], args['percentage'], args['seed'], args['model_seed'], args['update_mseed'], args['freeze'], lrs))

    print('leaner will be saved as %s'%bestLearner)
    return bestLearner,tempLearner

def define_save_path_los(args,lrs):
    bestLearner=os.path.join(data_dir,'models','los', '%s_%s_seed_%s_modelseed_%s_update_%s_freeze_%s_lr_%s-best' % (
    args['setting_name'], args['len_train'], args['seed'], args['model_seed'], args.get('update_mseed'), args.get('freeze'), lrs))

    tempLearner=os.path.join(data_dir,'models','los', '%s_%s_seed_%s_modelseed_%s_update_%s_freeze_%s_lr_%s-temp' % (
    args['setting_name'], args['len_train'], args['seed'], args['model_seed'], args.get('update_mseed'), args.get('freeze'), lrs))

    print('leaner will be saved as %s'%bestLearner)
    return bestLearner,tempLearner
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_model_type(args,args_json=None):
    if (args_json is not None) & (isinstance(args,argparse.Namespace)):
        iniFilePath = args.ini_file

        if 'logs/cpc/' in iniFilePath:
            matched_model_type = re.match(r".*logs/cpc/(\S*)/.*.ini",iniFilePath).group(1)
            if matched_model_type == args_json['model_type']:
                return matched_model_type
            elif args_json['model_type'] is None:
                return matched_model_type
            elif args_json['model_type'] is not None:
                raise NameError('your ini file path %s is incorrect'%iniFilePath)
        elif 'logs/cpc_full/' in iniFilePath:
            matched_model_type = re.match(r".*logs/cpc_full/(\S*)/.*.ini",iniFilePath).group(1)
            if matched_model_type == args_json['model_type']:
                return matched_model_type
            elif args_json['model_type'] is None:
                return matched_model_type
            elif args_json['model_type'] is not None:
                raise NameError('your ini file path %s is incorrect'%iniFilePath)
    else:
        iniFilePath = args.ini_file if isinstance(args,argparse.Namespace) else args            
        if re.match(r'.*CPAE_AT*', iniFilePath):
            return 'CPAE_AT'
        if re.match(r'.*CPAE_selfAT*', iniFilePath):
            return 'CPAE_selfAT'
        if re.match(r'.*CPAE1_LSTM_NO_BN.*', iniFilePath):
            return 'CPAE1_LSTM_NO_BN'
        elif re.match(r'.*CPLSTM3H.*',iniFilePath):
            return 'CPLSTM3H'
        elif re.match(r'.*CPAELSTM4_AT.*',iniFilePath):
            return 'CPAELSTM4_AT'
        elif re.match(r'.*CPAELSTM44_AT.*',iniFilePath):
            return 'CPAELSTM44_AT'
        elif re.match(r'.*CPAELSTM44_selfAT.*',iniFilePath):
            return 'CPAELSTM44_selfAT'
        elif re.match(r'.*CDCK3_S.*',iniFilePath):
            return 'CDCK3_S'
        elif re.match(r'.*CAE2_S.*',iniFilePath):
            return 'CAE2_S'
        elif re.match(r'.*CPLSTM3H.*',iniFilePath):
            return 'CPLSTM3H'
        elif re.match(r'.*CPLSTM4C.*',iniFilePath):
            return 'CPLSTM4C'
        elif re.match(r'.*CPLSTM4H.*',iniFilePath):
            return 'CPLSTM4H'
        elif re.match(r'.*CPLSTM2.*',iniFilePath):
            return 'CPLSTM2'
        elif re.match(r'.*CPLSTM3.*',iniFilePath):
            return 'CPLSTM3'
        elif re.match(r'.*CPLSTM4.*',iniFilePath):
            return 'CPLSTM4'
        elif re.match(r'.*CPAELSTM43.*',iniFilePath):
            return 'CPAELSTM43'
        elif re.match(r'.*CPAELSTM46.*',iniFilePath):
            return 'CPAELSTM46'
        elif re.match(r'.*CPAELSTM45.*',iniFilePath):
            return 'CPAELSTM45'
        elif re.match(r'.*CPAELSTM44.*',iniFilePath):
            return 'CPAELSTM44'
        elif re.match(r'.*CPAELSTM42.*',iniFilePath):
            return 'CPAELSTM42'
        elif re.match(r'.*CPAELSTM41.*',iniFilePath):
            return 'CPAELSTM41'
        elif re.match(r'.*CPLSTM.*',iniFilePath):
            return 'CPLSTM'
        elif re.match(r'.*CPAE1_NO_BN.*', iniFilePath):
            return 'CPAE1_NO_BN'
        elif re.match(r'.*CPAE4_NO_BN.*', iniFilePath):
            return 'CPAE4_NO_BN'
        elif re.match(r'.*CPAE1_S.*', iniFilePath):
            return 'CPAE1_S'
        elif re.match(r'.*CPAE4_S.*', iniFilePath):
            return 'CPAE4_S'
        elif re.match(r'.*CPAE7_S.*', iniFilePath):
            return 'CPAE7_S'
        elif re.match(r'.*CPAE1_LSTM.*', iniFilePath):
            return 'CPAE1_LSTM'
        elif re.match(r'.*CPAE1.*', iniFilePath):
            return 'CPAE1'
        elif re.match(r'.*CPAE2.*', iniFilePath):
            return 'CPAE2'
        elif re.match(r'.*CPAE3.*', iniFilePath):
            return 'CPAE3'
        elif re.match(r'.*CPAE4.*', iniFilePath):
            return 'CPAE4'
        elif re.match(r'.*CPAE7.*', iniFilePath):
            return 'CPAE7'
        elif re.match(r'.*CDCK2.*', iniFilePath):
            return 'CDCK2'
        elif re.match(r'.*CAE11.*', iniFilePath):
            return 'CAE11'
        elif re.match(r'.*CAE1.*', iniFilePath):
            return 'CAE1'
        elif re.match(r'.*CAE2.*', iniFilePath):
            return 'CAE2'
        elif re.match(r'.*AE1.*', iniFilePath):
            return 'AE1'
        elif re.match(r'.*AE2.*', iniFilePath):
            return 'AE2'
        elif re.match(r'.*CDCK3.*', iniFilePath):
            return 'CDCK3'
        elif re.match(r'.*end2end.*', iniFilePath):
            return 'Basic_Cnn'
        elif re.match(r'.*Basic_Cnn.*', iniFilePath):
            return 'Basic_Cnn'
        elif re.match(r'.*Basic_LSTM.*', iniFilePath):
            return 'Basic_LSTM'
        elif re.match(r'.*CAE_LSTM.*', iniFilePath):
            return 'CAE_LSTM'
        elif re.match(r'.*AE_LSTM.*', iniFilePath):
            return 'AE_LSTM'
        elif re.match(r'.*CPAETLSTM.*', iniFilePath):
            return 'CPAETLSTM'

        else:
            raise NameError('your ini file path %s is incorrect'%iniFilePath)

script_dir, data_dir, logging_dir = get_machine_dir()


def write_config(dic, filename):
    config = configparser.ConfigParser()
    strconfig = {item[0]: repr(item[1]) for item in dic.items()}
    config['DEFAULT'] = strconfig
    with open(filename, 'w') as f:
        config.write(f)


def write_mlp_performance(learner, args_json):
    y = learner.get_preds(ds_type=DatasetType.Valid)
    s_val = auroc_score(y[:][0], y[1])
    del y
    y = learner.get_preds(ds_type=DatasetType.Test)
    print(y)


def verbose(epoch, train_acc_ls, train_loss_ls, val_acc_ls, val_loss_ls, interval=25):
    train_acc_total = sum(train_acc_ls) / len(train_acc_ls)
    train_loss_average = sum(train_loss_ls) / len(train_loss_ls)
    val_acc_total = sum(val_acc_ls) / len(val_acc_ls)
    val_loss_average = sum(val_loss_ls) / len(val_loss_ls)
    print('\n =====================  epoch : {} ===================== \n'.format(epoch))
    for i in range(0, len(train_loss_ls), interval):
        print("  train_loss : {:.3f}  \t  train_accuracy: {:.3f}  ".format(train_loss_ls[i], train_acc_ls[i]))
    print("\n  average-train_loss : {:.3f}  \t  average-train_accuracy: {:.3f}  \n".format(train_acc_total,
                                                                                           train_loss_average))
    print("\n  average-val_loss : {:.3f}  \t  average-val_accuracy: {:.3f}  \n".format(val_acc_total, val_loss_average))
    print('\n ====================================================== \n '.format(epoch))


def get_config_dic(config):
    """
    load in config dict from `configparser` type `ini` file
    """
    cf = configparser.ConfigParser()
    cf.read(config)
    return convert_config_type(cf)


def convert_config_type(config):
    return {item[0]: eval(item[1]) for item in config.items('DEFAULT')}


def get_setting_name(string):
    base = os.path.basename(string)
    setting_name=os.path.splitext(base)[0]
    return setting_name


def load_json(file):
    with open(file) as f:
        c = f.readline()
    dic = json.loads(c)
    return dic


def save_task_listfile(out_dir, task='imp'):
    """Use this function to save task-specific data in the corresponding directory
    the task needs specific screening of cohort, see cohort selection in document
    """
    taskdic = {
        "imp": "in-hospital-mortality",
    }

    # ----------------------------
    # get cohort patient ID list from the task-specific data generated by benchmark workflow
    ref_dir = os.path.join(top_path, 'data', taskdic['imp'])
    g = []

    # read lists
    for filelist in ['train_listfile_raw.csv', 'val_listfile.csv', 'test_listfile.csv']:
        with open(os.path.join(ref_dir, filelist), 'r') as f:
            content = f.readlines()[1:]  # The first row is column names.
            for line in content:
                g.append(line.lstrip())

    # merge lists and save
    with open(os.path.join(out_dir, 'total_listfile.csv'), 'w') as f:
        for line in g:
            f.write(line)

    def write_delete(self):
        if not os.path.exists('delete.csv'):
            with open('delete.csv', 'w') as f:
                for i in self.deleteNeeded:
                    f.write('%d\n' % i)


def convert_to_dateformat(timeList):
    return list(map(dt.datetime.fromisoformat, timeList))


def setup_logs(save_dir, type, run_name):
    """

    :param save_dir:  the directory to set up logs
    :param type:  'model' for saving logs in 'logs/cpc'; 'imp' for saving logs in 'logs/imp'
    :param run_name:
    :return:logger
    """
    # initialize logger
    logger = logging.getLogger(type)
    logger.setLevel(logging.INFO)

    # create the logging file handler
    log_file = os.path.join(save_dir, run_name + ".log")
    fh = logging.FileHandler(log_file)

    # create the logging console handler
    ch = logging.StreamHandler()

    # format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# Don't run again.
# gen_variableList(overwirte=True)


###########Utils for network models
def makeFrameDimension(signals, frame_size, durations=None, stride=None):
    """Convert Inputs (shape: BatchSize*Length*Features) to
    Outputs:
        data: (shapes: BatchSize* num_frames * frame_size * features).
        frame_end: int tensor list, denote which frame is the final frame of the patient
                    (considering that there is padding)

    if stride is not specified, it is set as frame_size, 
    which means that there is no overlap between frames;
    if durations is None, it is set as the whole length of the input data for each sample.
    

    signals: (batch_size,channels,length,features)
 
    """
    batchsize = signals.shape[0]
    length = signals.shape[2]
    features = signals.shape[3]
    if stride is None: stride = frame_size
    frames = [signals[:, :, time:time + frame_size, :] for time in np.arange(0, length, stride)]
    num_frames = len(frames)
    frames = tuple(frames)
    frames = torch.cat(frames).float().to(device)
    frames = frames.view(batchsize, num_frames, frame_size, features)
    if isinstance(durations, list) or isinstance(durations, torch.FloatTensor):
        frame_end = tuple([(duration - frame_size) // stride + 1 for duration in durations])
        frame_end = torch.stack(frame_end).float().to(device)
    elif isinstance(durations, int):
        frame_end = frames.shape[1]
    elif durations is None:
        frame_end = frames.shape[1]
    else:
        raise TypeError('durations is not the expected type!!')

    return frames, frame_end


def key_to_npy_name(index):
    """index:'3_1' (patientid=3, episode=1)
        npy name:'3_episode1_tiemseries.csv.npy'
    """
    f = re.search(r'([0-9]+)_([0-9]+)', index)
    id = f.group(1)
    episode = f.group(2)
    return ('%d_episode%d_timeseries.csv.npy' % (id, episode))


def npy_name_to_key(file):
    temp = re.search(r'([0-9]+).*([0-9]+)_timeseries.csv.npy', file)
    id = temp.group(1)
    episode = temp.group(2)
    return '%s_%s' % (id, episode)


def npy_name_to_csv_name(file):
    temp = re.search(r'(.*).npy', file)
    return temp.group(1)


def csv_name_to_key(file):
    temp = re.search(r'([0-9]+).*([0-9]+)_timeseries.csv', file)
    id = temp.group(1)
    episode = temp.group(2)
    return '%s_%s' % (id, episode)


def csv_name_to_npy_name(file):
    return file + '.npy'


def correct_list_file(dir=os.path.join(top_path, 'mydata', 'in-hospital-mortality')):
    filelist = [f for f in os.listdir(dir) if 'npy' in f]
    namelist = [npy_name_to_csv_name(f) for f in filelist]
    with open(os.path.join(dir, 'total_listfile.csv')) as f, open(os.path.join(dir, 'temp.csv'), 'w') as g:
        lines = f.readlines()
        for line in lines:
            # print(line)
            if line.split(',')[0] in namelist:
                g.write(line)
    os.remove(os.path.join(dir, 'total_listfile.csv'))
    os.rename(os.path.join(dir, 'temp.csv'), os.path.join(dir, 'total_listfile.csv'))


# ======================


class Discretizer:
    def __init__(self, timestep=0.8, store_masks=True, impute_strategy='zero', start_time='zero', max_hours=48,
                 config_path=os.path.join(os.path.split(os.path.dirname(__file__))[0], 'ref/discretizer_config.json')):
        """ args
        :::input_strategy:'zero', 'normal_value', 'previous', 'next'
        """
        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy
        self._max_hours = max_hours

        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)  # the number of variables
        ts = [float(row[0]) for row in X]  # time points
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + eps  # to ensure time point is sorted

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if self._max_hours is not None:
            max_hours = self._max_hours
        elif end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        # the total # of time points
        N_bins = int((max_hours - eps) / self._timestep + 1.0 - eps)  # the same as `np.ceil` , project to bigger int

        #               ------|  full in begin/end pos  |-----
        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]  # begin position for each variable (after one-hot embedding)
        end_pos = [0 for i in range(N_channels)]  # end position for each variable (after one-hot embedding)

        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]  # dimensionality of the whole input vector

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)  # orignal shape
        original_value = [["" for j in range(N_channels)] for i in
                          range(N_bins)]  # initialize a matrix to save values before embedding
        total_data = 0
        unused_data = 0

        #                      -------------------------

        def write(data, bin_id, channel, value, begin_pos):
            """write the value of variable into data, for categorical variable, use one-hot encoding
            """
            channel_id = self._channel_to_id[channel]
            if self._is_categorical_channel[channel]:  # one-hot encoding
                category_id = self._possible_values[channel].index(value)  # the position of one hot
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:  # continuous variable
                data[bin_id, begin_pos[channel_id]] = float(value)

        #  --------------| full in `data` & `mask`|------------ 
        for row in X:
            # each row stand for each time point
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue  # ？
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                # each j is a variable
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                # because a number of time point point to the same bin_id
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1  # what exactly is `mask` for ? to label which data is valid ?

                write(data, bin_id, channel, row[j], begin_pos)  # value=row[j]
                original_value[bin_id][channel_id] = row[j]
        #       --------------------------------------------

        #      ===========|>>  impute missing values   <<|===========

        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]  # a list for each channel
            # len(self._id_to_channel) : number of variables
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        # mask indicate the data point is valided
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[
                                channel]  # if no previous value, fill with normal value
                        else:
                            imputed_value = prev_values[channel_id][-1]  # the latest value
                    write(data, bin_id, channel, imputed_value, begin_pos)  # use `write` to fill missing value

        # reverse bin_id , so that `previous` become `next`
        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins - 1, -1, -1):
                # reverse bin_id here
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        # * strategy 'zero' needs no operation for `data` is created by np.zeros 

        #              ============================================

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])

        # -----|  update statstics  |-----
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)  # percentage
        self._unused_data_sum += unused_data / (total_data + eps)  # percentage

        # !!!!! store mask will strongly change data
        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # -----|  create new header  |-----
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)

    def print_statistics(self):
        print("statistics of discretizer:")
        print("\tconverted {} examples".format(self._done_count))
        print("\taverage unused data = {:.2f} percent".format(100.0 * self._unused_data_sum / self._done_count))
        print("\taverage empty  bins = {:.2f} percent".format(100.0 * self._empty_bins_sum / self._done_count))


class Normalizer:
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]  # transfer to the format of list?

        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0

    def _feed_data(self, x):
        x = np.array(x)
        self._count += x.shape[0]
        if self._sum_x is None:
            self._sum_x = np.sum(x, axis=0)  # sum each column
            self._sum_sq_x = np.sum(x ** 2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x ** 2, axis=0)

    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(
                1.0 / (N - 1) * (self._sum_sq_x - 2.0 * self._sum_x * self._means + N * self._means ** 2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj={'means': self._means,
                             'stds': self._stds},
                        file=save_file,
                        protocol=2)

    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding='latin1')
            self._means = dct['means']
            self._stds = dct['stds']

    def transform(self, X):
        """ normalize each column (unit standard error, zero mean)
        """
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret


def filter_args(args, Model):
    model_args = {k: v for k, v in args.items() if
                  k in [p.name for p in inspect.signature(Model.__init__).parameters.values()]}
    return model_args
