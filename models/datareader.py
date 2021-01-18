import os, sys
from torch.nn.utils.rnn import pad_sequence

# add the top-level directory of this project to sys.path so that we can import modules without error
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import scipy, math, random
import sys
from matplotlib import pyplot as plt
import pandas as pd
import json
import datetime as dt
import models.utils as utils
from torchvision import transforms
from sklearn import preprocessing


# Overwrite the data.Dataset object of pyTorch,
# so that we can subsequently write dataloader




class Structure_IHM_Dataset(Dataset):
    r"""
    the `Dataset` that convert timeseries to data matrix of the same dimension
    """

    def __init__(self, data_dir, list_file,indices=None,age_min=None, percentage=1, total=None,random_seed=None, timestep=0.25, store_masks=True,
                 impute_strategy='previous', normalization=True, transforms=None, cont_channels=None, max_hours=48,
                 start_time='zero'):
        r""" data_dir: the folder containing timeseries.csv ; list_file : the csv file containing name of the timeseries
        ... Parameter for percentage: How much of the dataset you would like to include
        ... Parameter for random_seed: Control the seed for sampling the dataset
        ... Parameter for discretizer : timestep, store_masks,impute_strategy
        ... Parameter for normalizer :  normalizatio (boolen)
        """
        if os.path.exists(data_dir):
            self.data_dir = data_dir
        else:
            raise FileNotFoundError('can not find data_dir%s' % data_dir)

        if os.path.exists(list_file):
            self.list_file = pd.read_csv(list_file)
            if 'train' in list_file and (int(percentage) == 1 or percentage is None):
                pass
            elif 'train' in list_file and random_seed is None:
                self.list_file = self.list_file[self.list_file['%.3f' % percentage] == 1]
            elif 'train' in list_file and random_seed is not None:
                positives = list(self.list_file[self.list_file['y_true'] == 1].index.values.tolist())
                negatives = list(self.list_file[self.list_file['y_true'] == 0].index.values.tolist())
                # randomly sample labels at this percentage
                n = len(self.list_file)
                np.random.seed(random_seed)
                if percentage > 0.05:
                    samples = np.random.randint(n, size=int(n * percentage))
                else:
                    np.random.seed(random_seed)
                    positives = random.sample(positives, int(n * percentage / 2))
                    print(random_seed)
                    print(positives)
                    np.random.seed(random_seed)
                    negatives = random.sample(negatives, int(n * percentage / 2))
                    samples = [*positives, *negatives]
                self.list_file = self.list_file.iloc[samples, :]
        else:
            raise FileNotFoundError('can not find list file %s' % list_file)

        if indices is not None:
            self.list_file=self.list_file.iloc[indices,:]
        if total is not None:
            np.random.seed(random_seed)
            positives = list(self.list_file[self.list_file['y_true'] == 1].index.values.tolist())
            negatives = list(self.list_file[self.list_file['y_true'] == 0].index.values.tolist())
            n = len(self.list_file)
            if total>1000:
                samples = np.random.randint(n, size=int(n * percentage))
            else:

                positives = random.sample(positives, int(total / 2))
                negatives = random.sample(negatives, int(total / 2))
                samples = [*positives, *negatives]
            self.list_file = self.list_file.iloc[samples, :]

        if age_min is not None:
            self.list_file=self.list_file[(self.list_file['age']>age_min) & (self.list_file['age']<130) ]
        # to transform tensor
        self.transforms = transforms

        # key parameter for discretizer and normalizer
        self.timestep = timestep
        self.store_masks = store_masks
        self.impute_strategy = impute_strategy
        self.random_seed = random_seed
        # discretizer : structure temporal data
        self.discretizer = utils.Discretizer(self.timestep, self.store_masks, self.impute_strategy, max_hours=max_hours,
                                             start_time=start_time)

        # normalizer
        self.normalization = normalization
        if self.normalization:
            if cont_channels is None:
                self.cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
            self.normalizer = utils.Normalizer(fields=self.cont_channels)
            self.normalizer_state = 'ihm_ts:{:.2f}_impute:{}_start:zero_masks:{}_n:17903.normalizer'.format(
                self.timestep, self.impute_strategy, self.store_masks)

            main_path = os.path.split(os.path.dirname(__file__))[0]
            if os.path.exists(os.path.join(main_path, 'ref', self.normalizer_state)):
                self.normalizer.load_params(
                    os.path.join(main_path, 'ref', self.normalizer_state))  # normalizer need external parameter
            else:
                raise FileNotFoundError(
                    'Not normalizer for such condition, please run "create_normailer.sh" under script/')

    def __len__(self): # len(dataset)
        return self.list_file.shape[0]

    def read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self.data_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        x, header = self.discretizer.transform(np.stack(ret))  # raw data -> structure data
        if self.normalization:
            x = self.normalizer.transform(x)
        return (x, header)

    def __getitem__(self, index): # list[index]
        fn, y = self.list_file.iloc[index][['stay', 'y_true']]
        # except KeyError:
        #     print(self.list_file)
        y = torch.tensor(int(y), dtype=torch.int64)
        x = self.read_timeseries(fn)[0]
        x = torch.tensor(x, dtype=float).float()
        if self.transforms:
            x = self.transforms(x)
        return x, y


class Structure_LoS_tri_Dataset(Dataset):
    r"""
    the `Dataset` that convert timeseries to data matrix of the same dimension
    """

    def __init__(self, data_dir, list_file, percentage=1,age_min=None, random_seed=None, timestep=0.25, store_masks=True,
                 impute_strategy='previous', normalization=True, transforms=None, cont_channels=None, max_hours=48,
                 start_time='zero'):
        r""" data_dir: the folder containing timeseries.csv ; list_file : the csv file containing name of the timeseries
        ... Parameter for percentage: How much of the dataset you would like to include
        ... Parameter for random_seed: Control the seed for sampling the dataset
        ... Parameter for discretizer : timestep, store_masks,impute_strategy
        ... Parameter for normalizer :  normalizatio (boolen)
        """
        if os.path.exists(data_dir):
            self.data_dir = data_dir
        else:
            raise FileNotFoundError('can not find data_dir%s' % data_dir)

        if os.path.exists(list_file):
            self.list_file = pd.read_csv(list_file)
            if 'train' in list_file and (int(percentage) == 1 or percentage is None):
                pass
            elif 'train' in list_file and random_seed is None:
                self.list_file = self.list_file[self.list_file['%.3f' % percentage] == 1]
            elif 'train' in list_file and random_seed is not None:
                long_stay = list(self.list_file[self.list_file['y_true'] == 1].index.values.tolist())
                short_stay = list(self.list_file[self.list_file['y_true'] == 0].index.values.tolist())
                death=list(self.list_file[self.list_file['y_true'] == 2].index.values.tolist())
                # randomly sample labels at this percentage
                n = len(self.list_file)
                np.random.seed(random_seed)
                if percentage > 0.05:
                    samples = np.random.randint(n, size=int(n * percentage))
                else:
                    long_stay = random.sample(long_stay, int(n * percentage / 3))
                    print(long_stay)
                    short_stay = random.sample(short_stay, int(n * percentage / 3))
                    death = random.sample(death, int(n * percentage / 3))
                    samples = [*long_stay, *short_stay, *death]
                self.list_file = self.list_file.iloc[samples, :]

        else:
            raise FileNotFoundError('can not find list file %s' % list_file)

        if age_min is not None:
            self.list_file=self.list_file[(self.list_file['age']>age_min) & (self.list_file['age']<130) ]


        # to transform tensor
        self.transforms = transforms

        # key parameter for discretizer and normalizer
        self.timestep = timestep
        self.store_masks = store_masks
        self.impute_strategy = impute_strategy
        self.random_seed = random_seed
        # discretizer : structure temporal data
        self.discretizer = utils.Discretizer(self.timestep, self.store_masks, self.impute_strategy, max_hours=max_hours,
                                             start_time=start_time)

        # normalizer
        self.normalization = normalization
        if self.normalization:
            if cont_channels is None:
                self.cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
            self.normalizer = utils.Normalizer(fields=self.cont_channels)
            self.normalizer_state = 'ihm_ts:{:.2f}_impute:{}_start:zero_masks:{}_n:17903.normalizer'.format(
                self.timestep, self.impute_strategy, self.store_masks)

            main_path = os.path.split(os.path.dirname(__file__))[0]
            if os.path.exists(os.path.join(main_path, 'ref', self.normalizer_state)):
                self.normalizer.load_params(
                    os.path.join(main_path, 'ref', self.normalizer_state))  # normalizer need external parameter
            else:
                raise FileNotFoundError(
                    'Not normalizer for such condition, please run "create_normailer.sh" under script/')

    def __len__(self): # len(dataset)
        return self.list_file.shape[0]

    def read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self.data_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        x, header = self.discretizer.transform(np.stack(ret))  # raw data -> structure data
        if self.normalization:
            x = self.normalizer.transform(x)
        return (x, header)

    def __getitem__(self, index): # list[index]
        fn, y = self.list_file.iloc[index][['stay', 'y_true']]
        # except KeyError:
        #     print(self.list_file)
        y = torch.tensor(int(y), dtype=torch.int64)
        x = self.read_timeseries(fn)[0]
        x = torch.tensor(x, dtype=float).float()
        if self.transforms:
            x = self.transforms(x)
        return x, y



class Structure_LoS_bi_Dataset(Dataset):
    r"""
    the `Dataset` that convert timeseries to data matrix of the same dimension
    """

    def __init__(self, data_dir, list_file, percentage=1, random_seed=None, timestep=0.25, store_masks=True,
                 impute_strategy='previous', normalization=True, transforms=None, cont_channels=None, max_hours=48,
                 start_time='zero'):
        r""" data_dir: the folder containing timeseries.csv ; list_file : the csv file containing name of the timeseries
        ... Parameter for percentage: How much of the dataset you would like to include
        ... Parameter for random_seed: Control the seed for sampling the dataset
        ... Parameter for discretizer : timestep, store_masks,impute_strategy
        ... Parameter for normalizer :  normalizatio (boolen)
        """
        if os.path.exists(data_dir):
            self.data_dir = data_dir
        else:
            raise FileNotFoundError('can not find data_dir%s' % data_dir)

        if os.path.exists(list_file):
            self.list_file = pd.read_csv(list_file)
            self.list_file=self.list_file[self.list_file['y_true']<2].reset_index()
            if 'train' in list_file and (int(percentage) == 1 or percentage is None):
                pass
            elif 'train' in list_file and random_seed is None:
                self.list_file = self.list_file[self.list_file['%.3f' % percentage] == 1]
            elif 'train' in list_file and random_seed is not None:
                long_stay = list(self.list_file[self.list_file['y_true'] == 1].index.values.tolist())
                short_stay = list(self.list_file[self.list_file['y_true'] == 0].index.values.tolist())
                # randomly sample labels at this percentage
                n = len(self.list_file)
                np.random.seed(random_seed)
                if percentage > 0.05:
                    samples = np.random.randint(n, size=int(n * percentage))
                else:
                    np.random.seed(random_seed)
                    long_stay = random.sample(long_stay, int(n * percentage / 2))
                    print(random_seed)
                    print(long_stay)
                    np.random.seed(random_seed)
                    short_stay = random.sample(short_stay, int(n * percentage / 2))
                    samples = [*long_stay, *short_stay]
                self.list_file = self.list_file.iloc[samples, :]

        else:
            raise FileNotFoundError('can not find list file %s' % list_file)

        # to transform tensor
        self.transforms = transforms

        # key parameter for discretizer and normalizer
        self.timestep = timestep
        self.store_masks = store_masks
        self.impute_strategy = impute_strategy
        self.random_seed = random_seed
        # discretizer : structure temporal data
        self.discretizer = utils.Discretizer(self.timestep, self.store_masks, self.impute_strategy, max_hours=max_hours,
                                             start_time=start_time)

        # normalizer
        self.normalization = normalization
        if self.normalization:
            if cont_channels is None:
                self.cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
            self.normalizer = utils.Normalizer(fields=self.cont_channels)
            self.normalizer_state = 'ihm_ts:{:.2f}_impute:{}_start:zero_masks:{}_n:17903.normalizer'.format(
                self.timestep, self.impute_strategy, self.store_masks)

            main_path = os.path.split(os.path.dirname(__file__))[0]
            if os.path.exists(os.path.join(main_path, 'ref', self.normalizer_state)):
                self.normalizer.load_params(
                    os.path.join(main_path, 'ref', self.normalizer_state))  # normalizer need external parameter
            else:
                raise FileNotFoundError(
                    'Not normalizer for such condition, please run "create_normailer.sh" under script/')

    def __len__(self): # len(dataset)
        return self.list_file.shape[0]

    def read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self.data_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        x, header = self.discretizer.transform(np.stack(ret))  # raw data -> structure data
        if self.normalization:
            x = self.normalizer.transform(x)
        return (x, header)

    def __getitem__(self, index): # list[index]
        fn, y = self.list_file.iloc[index][['stay', 'y_true']]
        # except KeyError:
        #     print(self.list_file)
        y = torch.tensor(int(y), dtype=torch.int64)
        x = self.read_timeseries(fn)[0]
        x = torch.tensor(x, dtype=float).float()
        if self.transforms:
            x = self.transforms(x)
        return x, y

class Structure_LoS_reg_live_Dataset(Dataset):
    r"""
    the `Dataset` that convert timeseries to data matrix of the same dimension
    """

    def __init__(self, data_dir, list_file, percentage=1, random_seed=None, timestep=0.25, store_masks=True,
                 impute_strategy='previous', normalization=True, transforms=None, cont_channels=None, max_hours=48,
                 start_time='zero'):
        r""" data_dir: the folder containing timeseries.csv ; list_file : the csv file containing name of the timeseries
        ... Parameter for percentage: How much of the dataset you would like to include
        ... Parameter for random_seed: Control the seed for sampling the dataset
        ... Parameter for discretizer : timestep, store_masks,impute_strategy
        ... Parameter for normalizer :  normalizatio (boolen)
        """
        if os.path.exists(data_dir):
            self.data_dir = data_dir
        else:
            raise FileNotFoundError('can not find data_dir%s' % data_dir)
        n=len(list_file)
        if os.path.exists(list_file):
            self.list_file = pd.read_csv(list_file)
            self.list_file = self.list_file[self.list_file['y_true'] < 2].reset_index()
            if 'train' in list_file and (int(percentage) == 1 or percentage is None):
                pass
            elif 'train' in list_file and random_seed is not None:
                samples = np.random.randint(n, size=int(n * percentage))
                self.list_file = self.list_file.iloc[samples, :]

        else:
            raise FileNotFoundError('can not find list file %s' % list_file)

        # to transform tensor
        self.transforms = transforms

        # key parameter for discretizer and normalizer
        self.timestep = timestep
        self.store_masks = store_masks
        self.impute_strategy = impute_strategy
        self.random_seed = random_seed
        # discretizer : structure temporal data
        self.discretizer = utils.Discretizer(self.timestep, self.store_masks, self.impute_strategy, max_hours=max_hours,
                                             start_time=start_time)

        # normalizer
        self.normalization = normalization
        if self.normalization:
            if cont_channels is None:
                self.cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
            self.normalizer = utils.Normalizer(fields=self.cont_channels)
            self.normalizer_state = 'ihm_ts:{:.2f}_impute:{}_start:zero_masks:{}_n:17903.normalizer'.format(
                self.timestep, self.impute_strategy, self.store_masks)

            main_path = os.path.split(os.path.dirname(__file__))[0]
            if os.path.exists(os.path.join(main_path, 'ref', self.normalizer_state)):
                self.normalizer.load_params(
                    os.path.join(main_path, 'ref', self.normalizer_state))  # normalizer need external parameter
            else:
                raise FileNotFoundError(
                    'Not normalizer for such condition, please run "create_normailer.sh" under script/')

    def __len__(self): # len(dataset)
        return self.list_file.shape[0]

    def read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self.data_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        x, header = self.discretizer.transform(np.stack(ret))  # raw data -> structure data
        if self.normalization:
            x = self.normalizer.transform(x)
        return (x, header)

    def __getitem__(self, index): # list[index]
        fn, y = self.list_file.iloc[index][['stay', 'y_true_y']]
        # except KeyError:
        #     print(self.list_file)
        y = torch.tensor(y, dtype=float).float()
        x = self.read_timeseries(fn)[0]
        x = torch.tensor(x, dtype=float).float()
        if self.transforms:
            x = self.transforms(x)
        return x, y
class Structure_LoS_reg_all_Dataset(Dataset):
    r"""
    the `Dataset` that convert timeseries to data matrix of the same dimension
    """

    def __init__(self, data_dir, list_file, percentage=1, random_seed=None, timestep=0.25, store_masks=True,
                 impute_strategy='previous', normalization=True, transforms=None, cont_channels=None, max_hours=48,
                 start_time='zero'):
        r""" data_dir: the folder containing timeseries.csv ; list_file : the csv file containing name of the timeseries
        ... Parameter for percentage: How much of the dataset you would like to include
        ... Parameter for random_seed: Control the seed for sampling the dataset
        ... Parameter for discretizer : timestep, store_masks,impute_strategy
        ... Parameter for normalizer :  normalizatio (boolen)
        """
        if os.path.exists(data_dir):
            self.data_dir = data_dir
        else:
            raise FileNotFoundError('can not find data_dir%s' % data_dir)
        if os.path.exists(list_file):
            self.list_file = pd.read_csv(list_file)
            n = len(self.list_file)

            # self.list_file = self.list_file[self.list_file['y_true'] < 2].reset_index()
            if 'train' in list_file and (int(percentage) == 1 or percentage is None):
                pass
            elif 'train' in list_file and random_seed is not None:
                samples = np.random.randint(n, size=int(n * percentage))
                self.list_file = self.list_file.iloc[samples, :]

        else:
            raise FileNotFoundError('can not find list file %s' % list_file)

        # to transform tensor
        self.transforms = transforms

        # key parameter for discretizer and normalizer
        self.timestep = timestep
        self.store_masks = store_masks
        self.impute_strategy = impute_strategy
        self.random_seed = random_seed
        # discretizer : structure temporal data
        self.discretizer = utils.Discretizer(self.timestep, self.store_masks, self.impute_strategy, max_hours=max_hours,
                                             start_time=start_time)

        # normalizer
        self.normalization = normalization
        if self.normalization:
            if cont_channels is None:
                self.cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
            self.normalizer = utils.Normalizer(fields=self.cont_channels)
            self.normalizer_state = 'ihm_ts:{:.2f}_impute:{}_start:zero_masks:{}_n:17903.normalizer'.format(
                self.timestep, self.impute_strategy, self.store_masks)

            main_path = os.path.split(os.path.dirname(__file__))[0]
            if os.path.exists(os.path.join(main_path, 'ref', self.normalizer_state)):
                self.normalizer.load_params(
                    os.path.join(main_path, 'ref', self.normalizer_state))  # normalizer need external parameter
            else:
                raise FileNotFoundError(
                    'Not normalizer for such condition, please run "create_normailer.sh" under script/')

    def __len__(self): # len(dataset)
        return self.list_file.shape[0]

    def read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self.data_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        x, header = self.discretizer.transform(np.stack(ret))  # raw data -> structure data
        if self.normalization:
            x = self.normalizer.transform(x)
        return (x, header)

    def __getitem__(self, index): # list[index]
        fn, y = self.list_file.iloc[index][['stay', 'y_true_y']]
        # except KeyError:
        #     print(self.list_file)
        y = torch.tensor(y, dtype=float).float()
        x = self.read_timeseries(fn)[0]
        x = torch.tensor(x, dtype=float).float()
        if self.transforms:
            x = self.transforms(x)
        return x, y



class ToTensor(object):
    """Convert pandas dataframe object to tensor
    """

    def __call__(self, sample):
        patient, death, duration = sample['patient'], sample['death'], sample['duration']
        patient = torch.from_numpy(patient).type(torch.FloatTensor).float()
        death = torch.tensor(death, dtype=float).float()
        duration = torch.tensor(duration, dtype=float).float()
        return {'patient': patient,
                'death': death,
                'duration': duration
                }


class Normalize(object):
    """Normalize each sample by column"""

    def __call__(self, sample):
        scaler = preprocessing.StandardScaler()
        x, y = sample
        x = scaler.fit_transform(x)

        return x, y


class FillMissing(object):
    """Choose mode to fill the missing values. #TODO: Here I only wrote the "zero" method, which fills zero into all the missing positions.
    """

    def __init__(self, mode='zero'):
        self.mode = mode

    def __call__(self, sample):
        patient, death, duration = sample['patient'], sample['death'], sample['duration']
        if self.mode == 'zero':
            patient = np.nan_to_num(patient)
        return {'patient': patient, 'death': death, 'duration': duration}


class TrivialDataset(Dataset):
    def __init__(self, x, y, transforms=None):
        self.y = y
        x = np.reshape(x, (y.shape[0], -1))
        scaler = preprocessing.StandardScaler()
        x = scaler.fit_transform(x)
        self.x = x

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, item):
        x = self.x[item, :]
        y = self.y[item]

        return torch.from_numpy(x).float(), torch.tensor(y).long()


def split_Structure_dd(args_json, percentage=1, random_seed=None, out='loader'):
    train_percentage = percentage * 0.9
    val_percentage = percentage * 0.1
    train_seed = random_seed
    if random_seed is None:
        val_seed = train_seed
    else:
        val_seed = random_seed + 10
    if args_json['data_split'] == 'random':
        total_set = Structure_map_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay'),
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train', 'listfile.csv'))
        len_samples = len(total_set)
        args_json['len_samples'] = len_samples
        training_set, validation_set, test_set = utils.random_split(total_set,
                                                                    [int(len_samples * 9 / 11),
                                                                     int((len_samples) * 1 / 11),
                                                                     len_samples - int((len_samples) * 9 / 11) - int(
                                                                         len_samples / 11)]
                                                                    )
    else:
        training_set = Structure_map_Dataset(
            ts_end=48,
            data_dir=os.path.join(args_json['scr_data_path'], 'decompensation', 'train'),
            list_file=os.path.join(args_json['scr_data_path'], 'decompensation', 'train', 'listfile.csv'),
            percentage=train_percentage,
            random_seed=train_seed
        )
        validation_set = Structure_map_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'decompensation', 'train'),
            percentage=val_percentage,
            random_seed=val_seed,
            list_file=os.path.join(args_json['scr_data_path'], 'decompensation', 'train', 'listfile.csv'))
        test_set = Structure_map_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'decompensation', 'test'),
            list_file=os.path.join(args_json['scr_data_path'], 'decompensation', 'test', 'listfile.csv'))

    train_loader = DataLoader(training_set, batch_size=args_json['batch_size'],
                              num_workers=args_json['num_workers'])  # set shuffle to True
    validation_loader = DataLoader(validation_set, batch_size=args_json['batch_size'],
                                   num_workers=args_json['num_workers'])  # set shuffle to False
    test_loader = DataLoader(test_set, batch_size=args_json['batch_size'], num_workers=args_json['num_workers'])

    if out == 'loader':
        return train_loader, validation_loader, test_loader
    else:
        return training_set, validation_set, test_set


def split_Structure_los(args_json, percentage=1, num_per_class=None, total=None, random_seed=None, out='loader'):
    train_percentage = percentage * 0.9
    val_percentage = percentage * 0.1
    train_seed = random_seed
    if random_seed is None:
        val_seed = train_seed
    else:
        val_seed = random_seed + 10
    if args_json['data_split'] == 'random':
       raise NameError("data split is not suggested to set as random. Please set as 'bechmark' instead!")
    else:
        training_set = Structure_map_Dataset(
            num_per_class=num_per_class,
            total=total,
            ts_end=48,
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train'),
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train_listfile.csv'),
            percentage=train_percentage,
            random_seed=train_seed,
        )
        validation_set = Structure_map_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train'),
            percentage=val_percentage,
            random_seed=val_seed,
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'val_listfile.csv'))
        test_set = Structure_map_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'test'),
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'test', 'listfile.csv'))

    train_loader = DataLoader(training_set, batch_size=args_json['batch_size'],
                              num_workers=args_json['num_workers'])  # set shuffle to True
    validation_loader = DataLoader(validation_set, batch_size=args_json['batch_size'],
                                   num_workers=args_json['num_workers'])  # set shuffle to False
    test_loader = DataLoader(test_set, batch_size=args_json['batch_size'], num_workers=args_json['num_workers'])

    if out == 'loader':
        return train_loader, validation_loader, test_loader
    else:
        return training_set, validation_set, test_set



def split_Structure_los_bi(args_json, percentage=1,random_seed=None, out='loader'):
    train_seed = random_seed
    if random_seed is None:
        val_seed = train_seed
    else:
        val_seed = random_seed + 10
    if args_json['data_split'] == 'random':
       raise NameError("data split is not suggested to set as random. Please set as 'bechmark' instead!")
    else:
        training_set = Structure_LoS_bi_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train'),
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train_listfile_tri.csv'),
            random_seed=train_seed,
            percentage=percentage,
        )
        validation_set = Structure_LoS_bi_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train'),
            random_seed=val_seed,
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'val_listfile_tri.csv'))
        test_set = Structure_LoS_bi_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'test'),
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay',  'test_listfile_tri.csv'))

    train_loader = DataLoader(training_set, batch_size=args_json['batch_size'],
                              num_workers=args_json['num_workers'])  # set shuffle to True
    validation_loader = DataLoader(validation_set, batch_size=args_json['batch_size'],
                                   num_workers=args_json['num_workers'])  # set shuffle to False
    test_loader = DataLoader(test_set, batch_size=args_json['batch_size'], num_workers=args_json['num_workers'])

    if out == 'loader':
        return train_loader, validation_loader, test_loader
    else:
        return training_set, validation_set, test_set

def split_Structure_los_tri(args_json, percentage=1,age=None,random_seed=None, out='loader'):
    train_seed = random_seed
    if random_seed is None:
        val_seed = train_seed
    else:
        val_seed = random_seed + 10
    if args_json['data_split'] == 'random':
       raise NameError("data split is not suggested to set as random. Please set as 'bechmark' instead!")
    else:
        training_set = Structure_LoS_tri_Dataset(
            age_min=age,
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train'),
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train_listfile_tri_age.csv'),
            random_seed=train_seed,
            percentage=percentage,
        )
        validation_set = Structure_LoS_tri_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train'),
            random_seed=val_seed,
            age_min=age,
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'val_listfile_tri_age.csv'))
        test_set = Structure_LoS_tri_Dataset(
            age_min=age,
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'test'),
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay',  'test_listfile_tri_age.csv'))

    train_loader = DataLoader(training_set, batch_size=args_json['batch_size'],
                              num_workers=args_json['num_workers'])  # set shuffle to True
    validation_loader = DataLoader(validation_set, batch_size=args_json['batch_size'],
                                   num_workers=args_json['num_workers'])  # set shuffle to False
    test_loader = DataLoader(test_set, batch_size=args_json['batch_size'], num_workers=args_json['num_workers'])

    if out == 'loader':
        return train_loader, validation_loader, test_loader
    else:
        return training_set, validation_set, test_set



def split_Structure_los_reg_all(args_json, percentage=1,random_seed=None, out='loader'):
    train_seed = random_seed
    if random_seed is None:
        val_seed = train_seed
    else:
        val_seed = random_seed + 10
    if args_json['data_split'] == 'random':
       raise NameError("data split is not suggested to set as random. Please set as 'bechmark' instead!")
    else:
        training_set = Structure_LoS_reg_all_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train'),
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train_listfile_reg.csv'),
            random_seed=train_seed,
            percentage=percentage,
        )
        validation_set = Structure_LoS_reg_all_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'train'),
            random_seed=val_seed,
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'val_listfile_reg.csv'))
        test_set = Structure_LoS_reg_all_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'length-of-stay', 'test'),
            list_file=os.path.join(args_json['scr_data_path'], 'length-of-stay',  'test_listfile_reg.csv'))

    train_loader = DataLoader(training_set, batch_size=args_json['batch_size'],
                              num_workers=args_json['num_workers'])  # set shuffle to True
    validation_loader = DataLoader(validation_set, batch_size=args_json['batch_size'],
                                   num_workers=args_json['num_workers'])  # set shuffle to False
    test_loader = DataLoader(test_set, batch_size=args_json['batch_size'], num_workers=args_json['num_workers'])

    if out == 'loader':
        return train_loader, validation_loader, test_loader
    else:
        return training_set, validation_set, test_set


def split_Structure_Inhospital(args_json, percentage=1, random_seed=None, out='loader',age=None):
    """
    if 'loader',given args_json, return train_loader,validation_loader,test_loader
    if 'set',given args_json, return train_set,validation_set,test_set

    """
    print('random seed to split dataset is %s' % random_seed)
    if args_json['data_split'] == 'random':
        total_set = Structure_IHM_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'in-hospital-mortality'),
            list_file=os.path.join(utils.top_path, 'ref', 'all_listfile.csv'))
        len_samples = len(total_set)
        args_json['len_samples'] = len_samples
        training_set, validation_set, test_set = random_split(total_set,
                                                              [int(len_samples * 9 / 11),
                                                               int((len_samples) * 1 / 11),
                                                               len_samples - int((len_samples) * 9 / 11) - int(
                                                                   len_samples / 11)]
                                                              )
    else:

        training_set = Structure_IHM_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'in-hospital-mortality'),
            list_file=os.path.join(args_json['top_path'], 'ref', 'train_listfile_age.csv'),
            percentage=percentage,
            random_seed=random_seed,
            age_min=age
        )
        validation_set = Structure_IHM_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'in-hospital-mortality'),
            list_file=os.path.join(args_json['top_path'], 'ref', 'val_listfile_age.csv'),
        age_min=age)
        test_set = Structure_IHM_Dataset(
            data_dir=os.path.join(args_json['scr_data_path'], 'in-hospital-mortality'),
            list_file=os.path.join(args_json['top_path'], 'ref', 'test_listfile_age.csv'),
        age_min=age)

    train_loader = DataLoader(training_set, batch_size=args_json['batch_size'], shuffle=True, collate_fn=my_collate_fix,
                              num_workers=args_json['num_workers'])  # set shuffle to True
    validation_loader = DataLoader(validation_set, batch_size=args_json['batch_size'], shuffle=False,
                                   collate_fn=my_collate_fix,
                                   num_workers=args_json['num_workers'])  # set shuffle to False
    test_loader = DataLoader(test_set, batch_size=args_json['batch_size'], shuffle=False, collate_fn=my_collate_fix,
                             num_workers=args_json['num_workers'])

    if out == 'loader':
        return train_loader, validation_loader, test_loader
    else:
        return training_set, validation_set, test_set


def read_full_seq(args_json):
    training_set = Full_Length_Dataset(
        data_dir=os.path.join(args_json['scr_data_path'], 'pretrain', 'train'),
        list_file=os.path.join(args_json['scr_data_path'], 'pretrain', 'train_listfile.csv'),
        batch_size=args_json['batch_size'])
    validation_set = Full_Length_Dataset(
        data_dir=os.path.join(args_json['scr_data_path'], 'pretrain', 'train'),
        list_file=os.path.join(args_json['scr_data_path'], 'pretrain', 'val_listfile.csv'),
        batch_size=args_json['batch_size'])
    test_set = Full_Length_Dataset(
        data_dir=os.path.join(args_json['scr_data_path'], 'pretrain', 'test'),
        list_file=os.path.join(args_json['scr_data_path'], 'pretrain', 'test_listfile.csv'),
        batch_size=args_json['batch_size'])

    train_loader = DataLoader(training_set, batch_size=1,
                              num_workers=args_json['num_workers'],collate_fn=my_collate_fix_full)  # set shuffle to True
    validation_loader = DataLoader(validation_set, batch_size=1,
                                   num_workers=args_json['num_workers'],collate_fn=my_collate_fix_full)  # set shuffle to False
    test_loader = DataLoader(test_set, batch_size=1,
                             num_workers=args_json['num_workers'],collate_fn=my_collate_fix_full)

    return train_loader, validation_loader, test_loader


def my_collate_fix(batch):
    """Add paddings to samples in one batch to make sure that they have the same length.

       Args:
            Input:
            Output:
                data(tensor): a batch of data of patients with the same length
                labels(tensor): the labels of the data in this batch
                durations(tensor): the original lengths of the patients in the batch
        Shape:
            Input:
            Output:
                data: (batch_size,length,num_features)
                labels: (batch_size,)
                durations:(batch_size,)


    """
    if len(batch) == 1:
        return 1  # if batch size=1, it should be the last batch. we cannot compute the nce loss, so ignore this batch.
    else:
        return batch
def my_collate_fix_full(batch):
    """Add paddings to samples in one batch to make sure that they have the same length.

       Args:
            Input:
            Output:
                data(tensor): a batch of data of patients with the same length
                labels(tensor): the labels of the data in this batch
                durations(tensor): the original lengths of the patients in the batch
        Shape:
            Input:
            Output:
                data: (batch_size,length,num_features)
                labels: (batch_size,)
                durations:(batch_size,)


    """
    if len(batch[0]['data']) == 1:
        return 1  # if batch size=1, it should be the last batch. we cannot compute the nce loss, so ignore this batch.
    else:
        return {'data':torch.tensor(batch[0]['data']).float(),
                'length':batch[0]['length']}

# ---------------------------------------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------------------------------------

"""
dataset= InHospitalMortalityDataset (
    datadir=os.path.join(top_path,'mydata','in-hospital-mortality'),
    transforms=transforms.Compose([
        Normalize(), 
        FillMissing(), 
        ToTensor()
        ])
    )   

#print(dataset[1])
dataloader=DataLoader(dataset,batch_size=4,shuffle=True,num_workers=4)
"""


class Structure_map_Dataset(Dataset):
    def __init__(self, data_dir, list_file, steps=2000, ts_end=48, num_per_class=None, total=None, percentage=1,
                 random_seed=1, timestep=0.25, store_masks=True, impute_strategy='previous', max_hours=48,
                 normalization=True, transforms=None, cont_channels=None):

        """
        LOS Dataset that convert time series data into data chunk 
        three groups of parameters responsible for :

        - raw_data  
            ... data_dir : final dir of time series data
            ... list_file : absolute path of listfile.csv
            ... ts_end : int or None, defualt None can choose 48 

        - discretizer and normalizer 
            ... timestep : float,the time intervel to discretize data chunk
            ... store_masks : boolen ,  mask 
            ... impute_strategy : 'zero','previous'
            ... max_hours : float , defualt 48
            ... normalization : boolen 

        - torch transforms 
            ... transforms :  
        """

        #  read listfile
        if os.path.exists(list_file):
            self.list_file = pd.read_csv(list_file)
            self.list_file_total = self.list_file.period_length.values + self.list_file.y_true.values
            # self.list_file = self.list_file[self.list_file_total > 48]   # screen out episode that longer than 48 hours
        self._dataset_dir = data_dir
        self.total = total
        self.task = 'DC' if len(self.list_file.y_true.unique()) == 2 else 'LOS'
        self.num_per_class = num_per_class
        if ts_end:
            # pick up a subset of data whose period in ICU equals `ts_end`
            assert (isinstance(ts_end, int))
            self.list_file = self.list_file[self.list_file['period_length'] == ts_end].drop_duplicates(['stay'])

        if self.task == 'DD' and 'train' in list_file:
            positives = list(self.list_file[self.list_file['y_true'] == 1].index.values.tolist())
            negatives = list(self.list_file[self.list_file['y_true'] == 0].index.values.tolist())
            # randomly sample labels at this percentage
            n = len(self.list_file)
            np.random.seed(random_seed)
            if percentage > 0.05:
                samples = np.random.randint(0, n, size=int(n * percentage))
            else:
                np.random.seed(random_seed)
                positives = random.sample(positives, int(n * percentage / 2))
                np.random.seed(random_seed)
                negatives = random.sample(negatives, int(n * percentage / 2))
                samples = [*positives, *negatives]
            self.list_file = self.list_file.iloc[samples, :]

        # TODO percentage just be passed in,but previous results of 100% need be re-run

        if self.task == 'LOS':  # TODO random seed is not applied
            np.random.seed(random_seed)

            self.list_file['y_class'] = list(map(self.y2label, self.list_file['y_true'].to_list()))
            if 'train' in list_file:
                if num_per_class is None and total is None:
                    pass
                elif num_per_class is not None and num_per_class <= 100:
                    self.list_file = self.list_file.loc[
                                     self.draw_from_all_class(num_per_class, self.list_file, 'y_class'), :]
                # elif total < 5000:
                #     samples_base = self.draw_from_all_class(100, self.list_file, 'y_class')
                #     samples_rest = np.random.choice(self.list_file.index.values.tolist(), total - 1000)
                #     self.list_file = self.list_file.loc[[*samples_base, *samples_rest], :]
                else:
                    if total is None: total=num_per_class*10
                    samples_base = self.draw_from_all_class(10, self.list_file, 'y_class')
                    samples_rest = np.random.choice(self.list_file.index.values.tolist(), total - 100)
                    self.list_file = self.list_file.loc[[*samples_base, *samples_rest], :]

        #  initiate discretizer
        self.discretizer = utils.Discretizer(timestep, store_masks, impute_strategy, max_hours=max_hours)

        #  initiate normalizer 
        if cont_channels is None:
            self.cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
        self.normalizer = utils.Normalizer(fields=self.cont_channels)
        # load normalizer param
        self.normalizer_state = 'los_ts:{}_impute:previous_start:zero_masks:True_n:80000.normalizer'.format(timestep)
        self.normalizer_state = os.path.join(utils.top_path, 'ref', self.normalizer_state)
        assert (os.path.exists(self.normalizer_state))
        self.normalizer.load_params(self.normalizer_state)
        self.percentage = percentage
        # torch transforms
        self.transforms = transforms

    def __len__(self):
        return self.list_file.shape[0]

    def _read_timeseries(self, ts_filename, time_bound):
        # read X
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return np.stack(ret)

    def draw_from_one_class(self, num, df, column_to_draw, value):
        all = df[df[column_to_draw] == value].index.values.tolist()
        try:
            samples_index = np.random.choice(all, num, replace=False).tolist()
        except ValueError:
            samples_index = np.random.choice(all, num, replace=True).tolist()

        return samples_index

    def draw_from_all_class(self, num, df, column_to_draw):
        cls = df[column_to_draw].unique()
        index = []
        for i in range(len(cls)):
            index.extend(self.draw_from_one_class(num, df, column_to_draw, cls[i]))
        return index

    def y2label(self, y):
        # convert to  10 class 
        day = y // 24
        if day >= 8:
            day = 8 if (day in range(8, 14)) else 9
        return day

    def __getitem__(self, index):
        if self.task == 'LOS':
            fn, t, y, cls = self.list_file.iloc[index, :]
        else:
            fn, t, y = self.list_file.iloc[index, :]
        # convert remaining time to multiclass 

        # read X and transform
        X = self._read_timeseries(fn, t)
        X = self.discretizer.transform(X)[0]  # discretizer transform
        X = self.normalizer.transform(X)  # normalizer transform

        # from np.array -> tensor
        X = torch.tensor(X, dtype=float).float()
        y = torch.tensor(int(y), dtype=torch.int64)
        if self.task == 'LOS': y = torch.tensor(int(cls), dtype=torch.int64)
        if self.transforms:
            X = self.transforms(X)
        return X, y


class Structure_iter_Dataset(IterableDataset):
    def __init__(self, data_dir, list_file, batch_size, steps=2000, ts_end=None, percentage=1, random_seed=1,
                 timestep=0.25, store_masks=True,
                 impute_strategy='previous', normalization=True, transforms=None, cont_channels=None):

        self.current_index = 0
        self.batch_size = batch_size
        self.steps = steps
        self.n_examples = steps * batch_size
        self.chunk_size = min(1024, self.steps) * batch_size

        #  read listfile
        self._data_dir = data_dir
        if os.path.exists(list_file):
            self.list_file = pd.read_csv(list_file)
            self.list_file_total = self.list_file.period_length.values + self.list_file.y_true.values
            # self.list_file = self.list_file[self.list_file_total > 48]   # screen out episode that longer than 48 hours

        self.task = 'DC' if len(self.list_file.y_true.unique()) == 2 else 'LOS'

        if ts_end:
            # pick up a subset of data whose period in ICU equals `ts_end`
            assert (isinstance(ts_end, int))
            self.list_file = self.list_file[self.list_file['period_length'] == ts_end].drop_duplicates(['stay'])
        if 'train' in list_file and random_seed is None:
            self.list_file = self.list_file[self.list_file['%.3f' % percentage] == 1]
        elif 'train' in list_file and random_seed is not None:
            positives = list(self.list_file[self.list_file['y_true'] == 1].index.values.tolist())
            negatives = list(self.list_file[self.list_file['y_true'] == 0].index.values.tolist())
            # randomly sample labels at this percentage
            n = len(self.list_file)
            np.random.seed(random_seed)
            if percentage > 0.05:
                samples = np.random.randint(n, size=int(n * percentage))
            else:
                np.random.seed(random_seed)
                positives = random.sample(positives, int(n * percentage / 2))
                np.random.seed(random_seed)
                negatives = random.sample(negatives, int(n * percentage / 2))
                samples = [*positives, *negatives]
            self.list_file = self.list_file.iloc[samples, :]
        #  initiate discretizer
        self.discretizer = utils.Discretizer(timestep, store_masks, impute_strategy, max_hours=None)

        #  initiate normalizer
        if cont_channels is None:
            self.cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
        self.normalizer = utils.Normalizer(fields=self.cont_channels)
        # load normalizer param
        if ts_end:
            self.normalizer_state = 'ihm_ts:{:.2f}_impute:previous_start:zero_masks:True_n:17903.normalizer'.format(
                timestep)
        else:
            self.normalizer_state = 'los_ts:{}_impute:previous_start:zero_masks:True_n:80000.normalizer'.format(
                timestep)
        main_path = os.path.split(os.path.dirname(__file__))[0]
        if os.path.exists(os.path.join(main_path, 'ref', self.normalizer_state)):
            self.normalizer.load_params(
                os.path.join(main_path, 'ref', self.normalizer_state))

        # torch transforms
        self.transforms = transforms

    def __len__(self):
        return self.list_file.shape[0]

    def _read_timeseries(self, ts_filename, time_bound):
        # read X
        ret = []
        with open(os.path.join(self._data_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return np.stack(ret)

    def read_chunk(self, start, end):
        """
        read a lot of data
        """
        Xs = []
        ys = []
        ts = []

        for index in range(start, end):
            fn, t, y = self.list_file.iloc[index, :]
            # read X and transform
            X = self._read_timeseries(fn, t)
            X = self.discretizer.transform(X, end=t)[0]
            X = self.normalizer.transform(X)

            Xs.append(X)
            ys.append(y)
            ts.append(t)

        return Xs, ys, ts

    def y2label(self, y):
        # convert to  10 class
        day = y // 24
        if day >= 8:
            day = 8 if (day in range(8, 14)) else 9
        return day

    def sort_and_shuffle(self, data):
        """ Sort data by the length and then make batches and shuffle them.
            data is tuple (X1, X2, ..., Xn) all of them have the same length.
            Usually data = (X, y).
        """
        batch_size = self.batch_size
        assert len(data) >= 2
        data = list(zip(*data))

        random.shuffle(data)

        old_size = len(data)
        rem = old_size % batch_size
        head = data[:old_size - rem]
        tail = data[old_size - rem:]
        data = []

        head.sort(key=(lambda x: x[0].shape[0]))

        mas = [head[i: i + batch_size] for i in range(0, len(head), batch_size)]
        random.shuffle(mas)

        for x in mas:
            data += x
        data += tail

        data = list(zip(*data))
        return data

    def pad_zeros(self, arr, min_length=None):
        """
        `arr` is an array of `np.array`s

        The function appends zeros to every `np.array` in `arr`
        to equalize their first axis lenghts.
        """
        dtype = arr[0].dtype
        max_len = max([x.shape[0] for x in arr])
        ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in arr]
        if (min_length is not None) and ret[0].shape[0] < min_length:
            ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
                   for x in ret]
        return np.array(ret)

    def __iter__(self):

        #          ------------  multi processing  ------------
        worker_info = get_worker_info()

        if worker_info is None:  # single processing
            iter_start = 0
            iter_end = self.__len__()
        else:  # multi processing
            per_worker = int(math.ceil((self.__len__()) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.__len__())
        #                           ------------

        #            ------- the main part of iterator
        remaining = iter_end - iter_start  # not neccessary per_worker
        while remaining > 0:
            # TODO: read chunk
            chunk_size = remaining if (remaining < self.chunk_size) else self.chunk_size
            Xs, ys, ts = self.read_chunk(iter_start, iter_start + chunk_size)

            # shuffle sort length:
            Xs, ys, ts = self.sort_and_shuffle((Xs, ys, ts))
            if self.task == 'LOS':
                ys = [self.y2label(y) for y in ys]

            # update start
            iter_start += chunk_size
            remaining -= chunk_size

            # TODO: worker number
            for i in range(0, chunk_size, self.batch_size):
                X = self.pad_zeros(Xs[i:i + self.batch_size])
                y = np.array(ys[i:i + self.batch_size])
                batch_ts = ts[i:i + self.batch_size]
                batch_data = (X, y)

                yield batch_data


class Full_Length_Dataset(IterableDataset):
    """
    Based on Structure_iter_Dataset, read full length sequences and return the batch of data and the original unpadded lengths inside the batch
    """

    def __init__(self, data_dir, list_file, batch_size, steps=2000,
                 timestep=0.25, store_masks=True,
                 impute_strategy='previous', normalization=True, transforms=None, cont_channels=None):

        self.current_index = 0
        self.batch_size = batch_size
        self.steps = steps
        self.n_examples = steps * batch_size
        self.chunk_size = min(1024, self.steps) * batch_size

        #  read listfile
        self._data_dir = data_dir
        if os.path.exists(list_file):
            self.list_file = pd.read_csv(list_file)
        else: raise FileNotFoundError('The list_file path is incorrect %s'%list_file)

        self.task = 'DC' if len(self.list_file.y_true.unique()) == 2 else 'LOS'

        self.discretizer = utils.Discretizer(timestep, store_masks, impute_strategy, start_time='relative',max_hours=None)

        #  initiate normalizer
        if cont_channels is None:
            self.cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
        self.normalizer = utils.Normalizer(fields=self.cont_channels)
        # load normalizer param

        self.normalizer_state = 'los_ts:{}_impute:previous_start:zero_masks:True_n:80000.normalizer'.format(
                timestep)
        main_path = os.path.split(os.path.dirname(__file__))[0]
        if os.path.exists(os.path.join(main_path, 'ref', self.normalizer_state)):
            self.normalizer.load_params(
                os.path.join(main_path, 'ref', self.normalizer_state))

            # torch transforms
        self.transforms = transforms

    def __len__(self):
        return self.list_file.shape[0]

    def _read_timeseries(self, ts_filename):
        # read X
        ret = []
        with open(os.path.join(self._data_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                ret.append(np.array(mas))
        return np.stack(ret)

    def read_chunk(self, start, end):
        """
        read a lot of data
        """
        Xs = []
        ys = []

        for index in range(start, end):
            fn, y = self.list_file.iloc[index, :]
            # read X and transform
            X = self._read_timeseries(fn)
            X = self.discretizer.transform(X, end=None)[0]
            X = self.normalizer.transform(X)

            Xs.append(X)
            ys.append(y)

        return Xs, ys

    def y2label(self, y):
        # convert to  10 class
        day = y // 24
        if day >= 8:
            day = 8 if (day in range(8, 14)) else 9
        return day

    def sort_and_shuffle(self, data):
        """ Sort data by the length and then make batches and shuffle them.
            data is tuple (X1, X2, ..., Xn) all of them have the same length.
            Usually data = (X, y).
        """
        batch_size = self.batch_size
        assert len(data) >= 2
        data = list(zip(*data))

        random.shuffle(data)

        old_size = len(data)
        rem = old_size % batch_size
        head = data[:old_size - rem]
        tail = data[old_size - rem:]
        data = []

        head.sort(key=(lambda x: x[0].shape[0]))

        mas = [head[i: i + batch_size] for i in range(0, len(head), batch_size)]
        random.shuffle(mas)

        for x in mas:
            data += x
        data += tail

        data = list(zip(*data))
        return data

    def pad_zeros(self, arr, min_length=None):
        """
        `arr` is an array of `np.array`s

        The function appends zeros to every `np.array` in `arr`
        to equalize their first axis lenghts.
        """
        dtype = arr[0].dtype
        max_len = max([x.shape[0] for x in arr])
        ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in arr]
        if (min_length is not None) and ret[0].shape[0] < min_length:
            ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
                   for x in ret]
        return np.array(ret)

    def __iter__(self):

        #          ------------  multi processing  ------------
        worker_info = get_worker_info()

        if worker_info is None:  # single processing
            iter_start = 0
            iter_end = self.__len__()
        else:  # multi processing
            per_worker = int(math.ceil((self.__len__()) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.__len__())
        #                           ------------

        #            ------- the main part of iterator
        remaining = iter_end - iter_start  # not neccessary per_worker
        while remaining > 0:
            # TODO: read chunk
            chunk_size = remaining if (remaining < self.chunk_size) else self.chunk_size
            Xs, ys = self.read_chunk(iter_start, iter_start + chunk_size)

            # shuffle sort length:
            Xs, ys= self.sort_and_shuffle((Xs, ys))
            Ls = [len(x) for x in Xs]
            if self.task == 'LOS':
                ys = [self.y2label(y) for y in ys]

            # update start
            iter_start += chunk_size
            remaining -= chunk_size

            # TODO: worker number
            for i in range(0, chunk_size, self.batch_size):
                X = self.pad_zeros(Xs[i:i + self.batch_size])
                ls = Ls[i:i + self.batch_size]
                y = np.array(ys[i:i + self.batch_size])
                batch_data = {'data':X,
                              'length':ls}  # the length of each time series before packing

                yield batch_data

