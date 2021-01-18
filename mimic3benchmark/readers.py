from __future__ import absolute_import
from __future__ import print_function

import json
import os
import pickle
from sys import platform
import tensorflow.keras.backend as K
import numpy as np
import random
import pandas as pd
import tensorflow as tf
top_path='/home/shuying/cpcSurvival'
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
class Reader0(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class Reader(object):
    def __init__(self, dataset_dir, listfile=None,num_per_class=None,total=None,random_seed=1):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        self.random_seed=random_seed
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        self.list_file = pd.read_csv(listfile_path)

        if 'length-of-stay' in dataset_dir and ('train' in dataset_dir or 'train' in listfile):
            self.list_file['y_class'] = list(map(self.y2label, self.list_file['y_true'].to_list()))
            if num_per_class is None and total is None:
                pass
            elif num_per_class < 100:
                self.list_file = self.list_file.loc[self.draw_from_all_class(num_per_class, self.list_file, 'y_class'),:]
            elif total < 5000:
                samples_base = self.draw_from_all_class(100, self.list_file, 'y_class')
                np.random.seed(random_seed+1)
                samples_rest = np.random.choice(self.list_file.index.values.tolist(), total - 1000)
                self.list_file = self.list_file.iloc[[*samples_base, *samples_rest], :]
            else:
                samples_base = self.draw_from_all_class(10, self.list_file, 'y_class')
                np.random.seed(random_seed+1)
                samples_rest = np.random.choice(self.list_file.index.values.tolist(), total - 100)
                self.list_file = self.list_file.loc[[*samples_base, *samples_rest], :]
        self._listfile_header = self.list_file.columns
        self._data = self.list_file.values.tolist()
        self.n_samples=len(self.list_file)
    def y2label(self,y):
        # convert to  10 class
        day = y//24
        if day >= 8:
            day = 8 if (day in range(8,14)) else 9
        return day

    def draw_from_one_class(self, num, df, column_to_draw, value):
        all = df[df[column_to_draw] == value].index.values.tolist()
        np.random.seed(self.random_seed)
        try:
            samples_index = np.random.choice(all, num, replace=False).tolist()
        except ValueError:
            samples_index = np.random.choice(all, num, replace=True).tolist()

        return samples_index

    def draw_from_all_class(self,num,df,column_to_draw):
        cls=df[column_to_draw].unique()
        index=[]
        for i in range(len(cls)):
           index.extend(self.draw_from_one_class(num,df,column_to_draw,cls[i]))
        return index

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class DecompensationReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for decompensation prediction task.
        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), int(y)) for (x, t, y) in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
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
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Read the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Directory with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                Mortality within next 24 hours.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of examples (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name, t)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class InHospitalMortalityReader(Reader0):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        # self._data = [line.split(',') for line in self._data]   #private variable----why?
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class Structure_map_Dataset(object):
    def __init__(self, data_dir, list_file,steps=2000,ts_end=48, num_per_class=None,total=None,percentage=1,random_seed=1, timestep=0.25, store_masks=True,impute_strategy='previous', max_hours=48,normalization=True, transforms=None, cont_channels=None):

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
        self._dataset_dir=data_dir
        self.total=total
        self.task = 'DC' if len(self.list_file.y_true.unique()) == 2 else 'LOS'
        self.num_per_class=num_per_class
        if ts_end:
            # pick up a subset of data whose period in ICU equals `ts_end`
            assert(isinstance(ts_end,int))
            self.list_file = self.list_file[self.list_file['period_length'] == ts_end].drop_duplicates(['stay'])


        if self.task=='DD' and 'train' in list_file:
            positives = list(self.list_file[self.list_file['y_true'] == 1].index.values.tolist())
            negatives = list(self.list_file[self.list_file['y_true'] == 0].index.values.tolist())
            # randomly sample labels at this percentage
            n = len(self.list_file)
            np.random.seed(random_seed)
            if percentage > 0.05:
                samples = np.random.randint(0,n, size=int(n * percentage))
            else:
                np.random.seed(random_seed)
                positives = random.sample(positives, int(n * percentage / 2))
                np.random.seed(random_seed)
                negatives = random.sample(negatives, int(n * percentage / 2))
                samples = [*positives, *negatives]
            self.list_file = self.list_file.iloc[samples, :]

        # TODO percentage just be passed in,but previous results of 100% need be re-run

        if self.task=='LOS': #TODO random seed is not applied
            self.list_file['y_class'] = list(map(self.y2label, self.list_file['y_true'].to_list()))
            if 'train' in list_file:
                if num_per_class is None and total is None:
                    n = len(self.list_file)
                    np.random.seed(random_seed)
                    samples = np.random.randint(0, n, size=int(n * percentage))
                    self.list_file = self.list_file.iloc[samples, :]
                elif num_per_class<100 :
                    self.list_file=self.list_file.loc[self.draw_from_all_class(num_per_class,self.list_file,'y_class'),:]
                elif total<5000:
                    samples_base=self.draw_from_all_class(100,self.list_file,'y_class')
                    samples_rest=np.random.choice(self.list_file.index.values.tolist(),total-1000)
                    self.list_file=self.list_file.iloc[[*samples_base,*samples_rest],:]
                else:
                    samples_base = self.draw_from_all_class(10, self.list_file, 'y_class')
                    samples_rest = np.random.choice(self.list_file.index.values.tolist(), total - 100)
                    self.list_file = self.list_file.loc[[*samples_base, *samples_rest], :]

        #  initiate discretizer
        self.discretizer = Discretizer(timestep,store_masks,impute_strategy,max_hours=max_hours)

        #  initiate normalizer
        if cont_channels is None:
            self.cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
        self.normalizer = Normalizer(fields=self.cont_channels)
        # load normalizer param
        self.normalizer_state = 'los_ts:{}_impute:previous_start:zero_masks:True_n:80000.normalizer'.format(timestep)
        self.normalizer_state=os.path.join(top_path,'ref',self.normalizer_state)
        assert(os.path.exists(self.normalizer_state))
        self.normalizer.load_params(self.normalizer_state)
        self.percentage=percentage
        # torch transforms
        self.transforms = transforms

    def __len__(self):
        return self.list_file.shape[0]
    def get_x(self):
        fns, ts = self.list_file.iloc[:,0].values,self.list_file.iloc[:,1].values
        # convert remaining time to multiclass

        # read X and transform
        X = [self._read_timeseries(fns[i], ts[i]) for i in range(len(self.list_file))]
        X = [self.discretizer.transform(X[i])[0] for i in range(len(self.list_file))]  # discretizer transform
        X = [self.normalizer.transform(X[i]) for i in range(len(self.list_file))]  # normalizer transform
        # from np.array -> tensor
        X=np.array(X).reshape((len(self.list_file),192,76))
        return X
    def get_true(self):
        return pd.get_dummies(self.list_file['y_class']).to_numpy()


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


    def draw_from_one_class(self,num,df,column_to_draw,value):
        all=df[df[column_to_draw]==value].index.values.tolist()
        try:samples_index=np.random.choice(all,num,replace=False).tolist()
        except ValueError:samples_index=np.random.choice(all,num,replace=True).tolist()

        return samples_index

    def draw_from_all_class(self,num,df,column_to_draw):
        cls=df[column_to_draw].unique()
        index=[]
        for i in range(len(cls)):
           index.extend(self.draw_from_one_class(num,df,column_to_draw,cls[i]))
        return index




    def y2label(self,y):
        # convert to  10 class
        day = y//24
        if day >= 8:
            day = 8 if (day in range(8,14)) else 9
        return day

    def __getitem__(self,index):
        if self.task == 'LOS':
            fn,t,y,cls = self.list_file.iloc[index,:]
        else:
            fn, t, y = self.list_file.iloc[index, :]
        # convert remaining time to multiclass

        # read X and transform
        X = self._read_timeseries(fn,t)
        X = self.discretizer.transform(X)[0] #  discretizer transform
        X = self.normalizer.transform(X)  #  normalizer transform
        y=tf.one_hot(cls)
        # from np.array -> tensor
        if self.transforms:
            X = self.transforms(X)
        return X, y





class LengthOfStayReader(Reader):
    def __init__(self, dataset_dir, listfile=None,num_per_class=None,total=None,random_seed=1):
        """ Reader for length of stay prediction task.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile,num_per_class,total,random_seed)
        # self._data = [line.split(',') for line in self._data]
        # self._data = [(x, float(t), float(y)) for (x, t, y) in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
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
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : float
                Remaining time in ICU.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name, t)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class PhenotypingReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for phenotype classification task.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(mas[0], float(mas[1]), list(map(int, mas[2:]))) for mas in self._data]

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : array of ints
                Phenotype labels.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class MultitaskReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for multitask learning.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]

        def process_ihm(x):
            return list(map(int, x.split(';')))

        def process_los(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x)//2])), list(map(float, x[len(x)//2:])))

        def process_ph(x):
            return list(map(int, x.split(';')))

        def process_decomp(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x)//2])), list(map(int, x[len(x)//2:])))

        self._data = [(fname, float(t), process_ihm(ihm), process_los(los),
                       process_ph(pheno), process_decomp(decomp))
                      for fname, t, ihm, los, pheno, decomp in self._data]

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Return dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            ihm : array
                Array of 3 integers: [pos, mask, label].
            los : array
                Array of 2 arrays: [masks, labels].
            pheno : array
                Array of 25 binary integers (phenotype labels).
            decomp : array
                Array of 2 arrays: [masks, labels].
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": self._data[index][1],
                "ihm": self._data[index][2],
                "los": self._data[index][3],
                "pheno": self._data[index][4],
                "decomp": self._data[index][5],
                "header": header,
                "name": name}
