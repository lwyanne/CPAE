from __future__ import print_function
from torch.nn.utils.rnn import pack_padded_sequence

import inspect
import os, sys
import logging

# add the top-level directory of this project to sys.path so that we can import modules without error
from models.loss import Chimera_loss, record_loss, mask_where, mapping_where, mask_mapping_M

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logger = logging.getLogger("cpc")
import numpy as np
import torch
import torch.nn as nn
import math
from models.utils import *
from models.datareader import *
from sklearn.metrics import roc_auc_score
from fastai.callbacks import *
from fastai.tabular import *
from fastai import tabular
from models.optimizer import ScheduledOptim
from sklearn.metrics import cohen_kappa_score as kappa, mean_absolute_error as mad, roc_auc_score as auroc

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def auroc_score(input, target):
    input, target = input.cpu().numpy()[:, 1], target.cpu().numpy()
    return roc_auc_score(target, input)


class AUROC(tabular.Callback):
    """
    This is for output AUROC as a metric in fastai training process.
    This has a small but acceptable issue. #TODO
    """
    _order = -20  # Needs to run before the recorder

    def __init__(self, learn, **kwargs):
        self.learn = learn

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['AUROC'])

    def on_epoch_begin(self, **kwargs):
        self.output, self.target = [], []

    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            try:
                self.output.append(last_output)
            except AttributeError:
                self.output = []
            try:
                self.target.append(last_target)
            except AttributeError:
                self.target = []

    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output).cpu()
            target = torch.cat(self.target).cpu()
            preds = F.softmax(output, dim=1)
            metric = roc_auc_score(target, preds, multi_class='ovo')
            return add_metrics(last_metrics, [metric])


class biAUROC(tabular.Callback):
    """
    This is for output AUROC as a metric in fastai training process.
    This has a small but acceptable issue. #TODO
    """
    _order = -20  # Needs to run before the recorder

    def __init__(self, learn, **kwargs):
        self.learn = learn

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['AUROC'])

    def on_epoch_begin(self, **kwargs):
        self.output, self.target = [], []

    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            try:
                self.output.append(last_output)
            except AttributeError:
                self.output = []
            try:
                self.target.append(last_target)
            except AttributeError:
                self.target = []

    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output).cpu()
            target = torch.cat(self.target).cpu()
            preds = F.softmax(output, dim=1)
            metric = auroc_score(preds, target)
            return add_metrics(last_metrics, [metric])


class MAD(tabular.Callback):
    _order = -20

    def __init__(self, learn, **kwargs):
        self.learn = learn

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['MAD'])

    def on_epoch_begin(self, **kwargs):
        self.output, self.target = [], []

    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            try:
                self.output.append(last_output)
            except AttributeError:
                self.output = []
            try:
                self.target.append(last_target)
            except AttributeError:
                self.target = []

    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output)
            target = torch.cat(self.target)
            preds = torch.argmax(F.softmax(output, dim=1), dim=1, keepdim=False)
            metric = mean_absolute_error(preds, target)
            return add_metrics(last_metrics, [metric])


class CPclassifier(nn.Module):
    """
    Combine the CPC and MLP, to make it possible to fine-tune on the downstream task
    Note: Fine-tune is implemented via fastai learner.
    """

    def __init__(self, CPmodel, MLP, freeze=False):
        super(CPclassifier, self).__init__()
        self.CPmodel = CPmodel
        self.MLP = MLP
        if freeze:
            for param in self.CPmodel.parameters():
                param.requires_grad = False

    def forward(self, x):
        if 'CP' in self.CPmodel.__class__.__name__ or 'AE_LSTM' in self.CPmodel.__class__.__name__:
            x = self.CPmodel.get_reg_out(x)
        else:
            x = self.CPmodel.get_encode(x)
        x = self.MLP(x)
        return x


class CPAE1_S(nn.Module):
    def __init__(
            self,
            embedded_features,
            gru_out,
            conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8],
            time_step=30,
            n_points=192,
            n_features=76,
    ):
        self.embedded_features = embedded_features
        self.gru_out = gru_out
        self.conv_sizes = conv_sizes
        self.time_step = time_step
        # kernel_sizes=get_kernel_sizes() #TODO
        super(CPAE1_S, self).__init__()
        self.n_features = n_features
        # . If is int, uses the same padding in all boundaries.
        #  If a 4-tuple, uses (left ,right ,top ,bottom )
        self.channels = [n_features] + conv_sizes

        # the core part of model list
        self.sequential = lambda inChannel, outChannel: nn.Sequential(
            nn.ReflectionPad1d((0, 1)),
            nn.Conv1d(inChannel, outChannel, kernel_size=2, padding=0),
            nn.BatchNorm1d(outChannel),
            nn.ReLU(inplace=True)
        )

        # ** minded the length should be 1 element shorter than # of channels
        self.encoder = nn.ModuleList(
            [self.sequential(self.channels[i], self.channels[i + 1]) for i in range(len(conv_sizes))]
        ).to(device)

        self.decode_channels = self.channels[::-1]
        self.decoder = nn.ModuleList(
            [self.sequential(self.decode_channels[i], self.decode_channels[i + 1]) for i in range(len(conv_sizes))]
        ).to(device)
        self.linear = nn.Linear(self.conv_sizes[-1], self.embedded_features).to(device)

        self.Wk = nn.ModuleList([nn.Linear(self.gru_out, self.embedded_features) for i in range(self.time_step)]).to(
            device)

        # dim = 1 !!!
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.gru = nn.GRU(
            self.embedded_features,
            gru_out,
            num_layers=1,
            bidirectional=False,
            batch_first=True).to(device)
        self.beforeNCE = None

        # input shape: (N,C=1,n_points=192,n_features=76)
        # output shape: (N, C=sizes[-1], )
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(self._weights_init)
        # def relevant_points(n):

    def add_fcs(self, hidden=None):
        """
        This function will add FC layers to the embedded features and then compare the features after FC transformations.
        See NOTION for illustration.
        :param hidden: a list of hidden sizes per layer. For example:[100,100]. If no value is passed, it will be set
                    as [n_embedded_features,n_embedded_features]
        :return: None
        """
        n = self.embedded_features
        if hidden is None:
            self.fcs = nn.Sequential(
                nn.Linear(n, n),
                nn.ReLU(inplace=True),
                nn.Linear(n, n)
            )
        else:
            if type(hidden) != list:
                hidden = list(hidden)
            layers = []
            for i, j in zip([n] + hidden, hidden + [n]):
                layers.append(nn.Linear(i, j))
                layers.append(nn.ReLU(inplace=True))
            layers.pop()  # We do not want Relu at the last layer

            self.fcs = nn.Sequential(*layers).to(device)
        self.beforeNCE = True

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size, use_gpu=True):
        return torch.zeros(1, batch_size, self.gru_out).to(device)

    def encode(self, x):
        for i in range(len(self.encoder)):  # input shape:  (N,n_features=76,n_points=192)
            x = self.encoder[i](x)
        return x  # output shape: (N,n_features=8,n_points=192)

    def decode(self, x):
        for i in range(len(self.decoder)):  # input shape:   (N,n_features=8,n_points=192)
            x = self.decoder[i](x)
        return x  # output shape:  (N,n_points=192,n_features=76)

    def recurrent(self, zt):
        '''
        GRU RNN
        '''
        batch_size = self.batch_size
        # output shape: (N,  n_frames, features,1)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.gru(zt, hidden)
        return output, hidden

    def gru_to_ct(self, zt):
        '''
        return the last time_step of GRU result
        '''
        output, hidden = self.recurrent(zt)
        c_t = output[:, -1, :].view(self.batch_size, self.gru_out)
        return c_t, hidden

    def compute_nce(self, encode_samples, pred):
        '''
        -----------------------------------------------------------------------------------
                        --------------Calculate NCE loss--------------
        -----------------------------------------------------------------------------------
        ...argument:
        ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
        ......pred :            Wk[i]( C_t )
        '''
        nce = 0  # average over time_step and batch
        for i in np.arange(0, self.time_step):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            # print(total)
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.batch_size * self.time_step
        accuracy = 1. * correct.item() / self.batch_size

        return nce, accuracy

    def get_reg_out(self, x):
        self.batch_size = x.shape[0]

        x = x.squeeze(1).transpose(1, 2)
        self.n_frames = x.shape[2]

        z = self.encode(x).transpose(1, 2)
        z = self.linear(z)

        forward_seq = z[:, :, :]
        c_t, hidden = self.gru_to_ct(forward_seq)
        return c_t

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.encode(x).transpose(1, 2)  # z: (batch, n_time, conv[-1])
        d = self.decode(z.transpose(1, 2))

        self.batch_size = x.shape[0]
        self.n_frames = x.shape[2]

        # make change to here
        # t_samples should at least start from 30
        t_samples = torch.randint(low=self.time_step, high=self.n_frames - self.time_step - 1, size=(1,)).long().to(
            device)
        #
        encode_samples = torch.empty((self.time_step, self.batch_size, self.embedded_features)).float().to(
            device)  # e.g.
        # size

        z = self.linear(z)
        for i in np.arange(1, self.time_step + 1):
            encode_samples[i - 1, :, :] = z[:, int(t_samples) + i, :]

        forward_seq = z[:, :int(t_samples) + 1, :]

        c_t, hidden = self.gru_to_ct(forward_seq)

        pred = torch.empty((self.time_step, self.batch_size, self.embedded_features)).float().to(device)
        for i in np.arange(0, self.time_step):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        if self.beforeNCE:  # ADD FC layers
            pred = self.fcs(pred)
            encode_samples = self.fcs(encode_samples)
        # d = self.decode(pred.transpose(1,2).transpose(0,2))
        nce, accuracy = self.compute_nce(encode_samples, pred)

        return d, nce, accuracy


class CPAE1_NO_BN(nn.Module):
    def __init__(
            self,
            embedded_features,
            gru_out,
            conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8],
            time_step=30,
            n_points=192,
            n_features=76,
    ):
        self.embedded_features = embedded_features
        self.gru_out = gru_out
        self.conv_sizes = conv_sizes
        self.time_step = time_step
        # kernel_sizes=get_kernel_sizes() #TODO
        super(CPAE1_NO_BN, self).__init__()
        self.n_features = n_features
        # . If is int, uses the same padding in all boundaries.
        #  If a 4-tuple, uses (left ,right ,top ,bottom )
        self.channels = [n_features] + conv_sizes

        # the core part of model list
        self.sequential = lambda inChannel, outChannel: nn.Sequential(
            nn.ReflectionPad1d((0, 1)),
            nn.Conv1d(inChannel, outChannel, kernel_size=2, padding=0),
            # nn.BatchNorm1d(outChannel),
            nn.ReLU(inplace=True)
        )

        # ** minded the length should be 1 element shorter than # of channels
        self.encoder = nn.ModuleList(
            [self.sequential(self.channels[i], self.channels[i + 1]) for i in range(len(conv_sizes))]
        ).to(device)

        self.decode_channels = self.channels[::-1]
        self.decoder = nn.ModuleList(
            [self.sequential(self.decode_channels[i], self.decode_channels[i + 1]) for i in range(len(conv_sizes))]
        ).to(device)
        self.linear = nn.Linear(self.conv_sizes[-1], self.embedded_features).to(device)

        self.Wk = nn.ModuleList([nn.Linear(self.gru_out, self.embedded_features) for i in range(self.time_step)]).to(
            device)

        # dim = 1 !!!
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.gru = nn.GRU(
            self.embedded_features,
            gru_out,
            num_layers=1,
            bidirectional=False,
            batch_first=True).to(device)
        self.beforeNCE = None

        # input shape: (N,C=1,n_points=192,n_features=76)
        # output shape: (N, C=sizes[-1], )
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(self._weights_init)
        # def relevant_points(n):

    def add_fcs(self, hidden=None):
        """
        This function will add FC layers to the embedded features and then compare the features after FC transformations.
        See NOTION for illustration.
        :param hidden: a list of hidden sizes per layer. For example:[100,100]. If no value is passed, it will be set
                    as [n_embedded_features,n_embedded_features]
        :return: None
        """
        n = self.embedded_features
        if hidden is None:
            self.fcs = nn.Sequential(
                nn.Linear(n, n),
                nn.ReLU(inplace=True),
                nn.Linear(n, n)
            )
        else:
            if type(hidden) != list:
                hidden = list(hidden)
            layers = []
            for i, j in zip([n] + hidden, hidden + [n]):
                layers.append(nn.Linear(i, j))
                layers.append(nn.ReLU(inplace=True))
            layers.pop()  # We do not want Relu at the last layer

            self.fcs = nn.Sequential(*layers).to(device)
        self.beforeNCE = True

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size, use_gpu=True):
        return torch.zeros(1, batch_size, self.gru_out).to(device)

    def encode(self, x):
        for i in range(len(self.encoder)):  # input shape:  (N,n_features=76,n_points=192)
            x = self.encoder[i](x)
        return x  # output shape: (N,n_features=8,n_points=192)

    def decode(self, x):
        for i in range(len(self.decoder)):  # input shape:   (N,n_features=8,n_points=192)
            x = self.decoder[i](x)
        return x  # output shape:  (N,n_points=192,n_features=76)

    def recurrent(self, zt):
        '''
        GRU RNN
        '''
        batch_size = self.batch_size
        # output shape: (N,  n_frames, features,1)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.gru(zt, hidden)
        return output, hidden

    def gru_to_ct(self, zt):
        '''
        return the last time_step of GRU result
        '''
        output, hidden = self.recurrent(zt)
        c_t = output[:, -1, :].view(self.batch_size, self.gru_out)
        return c_t, hidden

    def compute_nce(self, encode_samples, pred):
        '''
        -----------------------------------------------------------------------------------
                        --------------Calculate NCE loss--------------
        -----------------------------------------------------------------------------------
        ...argument:
        ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
        ......pred :            Wk[i]( C_t )
        '''
        nce = 0  # average over time_step and batch
        for i in np.arange(0, self.time_step):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            # print(total)
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.batch_size * self.time_step
        accuracy = 1. * correct.item() / self.batch_size

        return nce, accuracy

    def get_reg_out(self, x):
        self.batch_size = x.shape[0]

        x = x.squeeze(1).transpose(1, 2)
        self.n_frames = x.shape[2]

        z = self.encode(x).transpose(1, 2)
        z = self.linear(z)

        forward_seq = z[:, :, :]
        c_t, hidden = self.gru_to_ct(forward_seq)
        return c_t

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.encode(x).transpose(1, 2)  # z: (batch, n_time, conv[-1])
        d = self.decode(z.transpose(1, 2))

        self.batch_size = x.shape[0]
        self.n_frames = x.shape[2]

        # make change to here
        # t_samples should at least start from 30
        t_samples = torch.randint(low=self.time_step, high=self.n_frames - self.time_step - 1, size=(1,)).long().to(
            device)
        #
        encode_samples = torch.empty((self.time_step, self.batch_size, self.embedded_features)).float().to(
            device)  # e.g.
        # size

        z = self.linear(z)
        for i in np.arange(1, self.time_step + 1):
            encode_samples[i - 1, :, :] = z[:, int(t_samples) + i, :]

        forward_seq = z[:, :int(t_samples) + 1, :]

        c_t, hidden = self.gru_to_ct(forward_seq)

        pred = torch.empty((self.time_step, self.batch_size, self.embedded_features)).float().to(device)
        for i in np.arange(0, self.time_step):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        # d = self.decode(pred.transpose(1,2).transpose(0,2))
        nce, accuracy = self.compute_nce(encode_samples, pred)

        return d, nce, accuracy


class CPAE1_LSTM(nn.Module):
    def __init__(
            self,
            embedded_features,
            gru_out,
            conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8],
            time_step=30,
            n_points=192,
            n_features=76,
    ):
        self.embedded_features = embedded_features
        self.gru_out = gru_out
        self.conv_sizes = conv_sizes
        self.time_step = time_step
        # kernel_sizes=get_kernel_sizes() #TODO
        super(CPAE1_LSTM, self).__init__()
        self.n_features = n_features
        # . If is int, uses the same padding in all boundaries.
        #  If a 4-tuple, uses (left ,right ,top ,bottom )
        self.channels = [n_features] + conv_sizes

        # the core part of model list
        self.sequential = lambda inChannel, outChannel: nn.Sequential(
            nn.ReflectionPad1d((0, 1)),
            nn.Conv1d(inChannel, outChannel, kernel_size=2, padding=0),
            nn.BatchNorm1d(outChannel),
            nn.ReLU(inplace=True)
        )

        # ** minded the length should be 1 element shorter than # of channels
        self.encoder = nn.ModuleList(
            [self.sequential(self.channels[i], self.channels[i + 1]) for i in range(len(conv_sizes))]
        ).to(device)

        self.decode_channels = self.channels[::-1]
        self.decoder = nn.ModuleList(
            [self.sequential(self.decode_channels[i], self.decode_channels[i + 1]) for i in range(len(conv_sizes))]
        ).to(device)
        self.linear = nn.Linear(self.conv_sizes[-1], self.embedded_features).to(device)

        self.Wk = nn.ModuleList([nn.Linear(self.gru_out, self.embedded_features) for i in range(self.time_step)]).to(
            device)

        # dim = 1 !!!
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.gru = nn.LSTM(
            self.embedded_features,
            hidden_size=gru_out,
            num_layers=2,
            bidirectional=False,
            batch_first=True).to(device)
        self.beforeNCE = None

        # input shape: (N,C=1,n_points=192,n_features=76)
        # output shape: (N, C=sizes[-1], )
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(self._weights_init)
        # def relevant_points(n):

    def add_fcs(self, hidden=None):
        """
        This function will add FC layers to the embedded features and then compare the features after FC transformations.
        See NOTION for illustration.
        :param hidden: a list of hidden sizes per layer. For example:[100,100]. If no value is passed, it will be set
                    as [n_embedded_features,n_embedded_features]
        :return: None
        """
        n = self.embedded_features
        if hidden is None:
            self.fcs = nn.Sequential(
                nn.Linear(n, n),
                nn.ReLU(inplace=True),
                nn.Linear(n, n)
            )
        else:
            if type(hidden) != list:
                hidden = list(hidden)
            layers = []
            for i, j in zip([n] + hidden, hidden + [n]):
                layers.append(nn.Linear(i, j))
                layers.append(nn.ReLU(inplace=True))
            layers.pop()  # We do not want Relu at the last layer

            self.fcs = nn.Sequential(*layers).to(device)
        self.beforeNCE = True

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size, use_gpu=True):
        return torch.zeros(1, batch_size, self.gru_out).to(device)

    def encode(self, x):
        for i in range(len(self.encoder)):  # input shape:  (N,n_features=76,n_points=192)
            x = self.encoder[i](x)
        return x  # output shape: (N,n_features=8,n_points=192)

    def decode(self, x):
        for i in range(len(self.decoder)):  # input shape:   (N,n_features=8,n_points=192)
            x = self.decoder[i](x)
        return x  # output shape:  (N,n_points=192,n_features=76)

    def recurrent(self, zt):
        '''
        GRU RNN
        '''
        batch_size = self.batch_size
        # output shape: (N,  n_frames, features,1)
        hidden = self.init_hidden(batch_size)
        hidden = torch.cat((hidden, hidden), dim=0)
        hidden = (hidden, hidden)
        output, hidden = self.gru(zt, hidden)
        return output, hidden

    def gru_to_ct(self, zt):
        '''
        return the last time_step of GRU result
        '''
        output, hidden = self.recurrent(zt)
        c_t = output[:, -1, :].view(self.batch_size, self.gru_out)
        return c_t, hidden

    def compute_nce(self, encode_samples, pred):
        '''
        -----------------------------------------------------------------------------------
                        --------------Calculate NCE loss--------------
        -----------------------------------------------------------------------------------
        ...argument:
        ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
        ......pred :            Wk[i]( C_t )
        '''
        nce = 0  # average over time_step and batch
        for i in np.arange(0, self.time_step):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            # print(total)
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.batch_size * self.time_step
        accuracy = 1. * correct.item() / self.batch_size

        return nce, accuracy

    def get_reg_out(self, x):
        self.batch_size = x.shape[0]

        x = x.squeeze(1).transpose(1, 2)
        self.n_frames = x.shape[2]

        z = self.encode(x).transpose(1, 2)
        z = self.linear(z)

        forward_seq = z[:, :, :]
        c_t, hidden = self.gru_to_ct(forward_seq)
        return c_t

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.encode(x).transpose(1, 2)  # z: (batch, n_time, conv[-1])
        d = self.decode(z.transpose(1, 2))

        self.batch_size = x.shape[0]
        self.n_frames = x.shape[2]

        # make change to here
        # t_samples should at least start from 30
        t_samples = torch.randint(low=self.time_step, high=self.n_frames - self.time_step - 1, size=(1,)).long().to(
            device)
        #
        encode_samples = torch.empty((self.time_step, self.batch_size, self.embedded_features)).float().to(
            device)  # e.g.
        # size

        z = self.linear(z)
        for i in np.arange(1, self.time_step + 1):
            encode_samples[i - 1, :, :] = z[:, int(t_samples) + i, :]

        forward_seq = z[:, :int(t_samples) + 1, :]

        c_t, hidden = self.gru_to_ct(forward_seq)

        pred = torch.empty((self.time_step, self.batch_size, self.embedded_features)).float().to(device)
        for i in np.arange(0, self.time_step):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        # d = self.decode(pred.transpose(1,2).transpose(0,2))
        nce, accuracy = self.compute_nce(encode_samples, pred)

        return d, nce, accuracy


class CPAE1_LSTM_NO_BN(nn.Module):
    def __init__(
            self,
            embedded_features,
            gru_out,
            conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8],
            time_step=30,
            n_points=192,
            n_features=76,
    ):
        self.embedded_features = embedded_features
        self.gru_out = gru_out
        self.conv_sizes = conv_sizes
        self.time_step = time_step
        # kernel_sizes=get_kernel_sizes() #TODO
        super(CPAE1_LSTM_NO_BN, self).__init__()
        self.n_features = n_features
        # . If is int, uses the same padding in all boundaries.
        #  If a 4-tuple, uses (left ,right ,top ,bottom )
        self.channels = [n_features] + conv_sizes

        # the core part of model list
        self.sequential = lambda inChannel, outChannel: nn.Sequential(
            nn.ReflectionPad1d((0, 1)),
            nn.Conv1d(inChannel, outChannel, kernel_size=2, padding=0),
            # nn.BatchNorm1d(outChannel),
            nn.ReLU(inplace=True)
        )

        # ** minded the length should be 1 element shorter than # of channels
        self.encoder = nn.ModuleList(
            [self.sequential(self.channels[i], self.channels[i + 1]) for i in range(len(conv_sizes))]
        ).to(device)

        self.decode_channels = self.channels[::-1]
        self.decoder = nn.ModuleList(
            [self.sequential(self.decode_channels[i], self.decode_channels[i + 1]) for i in range(len(conv_sizes))]
        ).to(device)
        self.linear = nn.Linear(self.conv_sizes[-1], self.embedded_features).to(device)

        self.Wk = nn.ModuleList([nn.Linear(self.gru_out, self.embedded_features) for i in range(self.time_step)]).to(
            device)

        # dim = 1 !!!
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.gru = nn.LSTM(
            self.embedded_features,
            hidden_size=gru_out,
            num_layers=2,
            bidirectional=False,
            batch_first=True).to(device)
        self.beforeNCE = None

        # input shape: (N,C=1,n_points=192,n_features=76)
        # output shape: (N, C=sizes[-1], )
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(self._weights_init)
        # def relevant_points(n):

    def add_fcs(self, hidden=None):
        """
        This function will add FC layers to the embedded features and then compare the features after FC transformations.
        See NOTION for illustration.
        :param hidden: a list of hidden sizes per layer. For example:[100,100]. If no value is passed, it will be set
                    as [n_embedded_features,n_embedded_features]
        :return: None
        """
        n = self.embedded_features
        if hidden is None:
            self.fcs = nn.Sequential(
                nn.Linear(n, n),
                nn.ReLU(inplace=True),
                nn.Linear(n, n)
            )
        else:
            if type(hidden) != list:
                hidden = list(hidden)
            layers = []
            for i, j in zip([n] + hidden, hidden + [n]):
                layers.append(nn.Linear(i, j))
                layers.append(nn.ReLU(inplace=True))
            layers.pop()  # We do not want Relu at the last layer

            self.fcs = nn.Sequential(*layers).to(device)
        self.beforeNCE = True

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size, use_gpu=True):
        return torch.zeros(1, batch_size, self.gru_out).to(device)

    def encode(self, x):
        for i in range(len(self.encoder)):  # input shape:  (N,n_features=76,n_points=192)
            x = self.encoder[i](x)
        return x  # output shape: (N,n_features=8,n_points=192)

    def decode(self, x):
        for i in range(len(self.decoder)):  # input shape:   (N,n_features=8,n_points=192)
            x = self.decoder[i](x)
        return x  # output shape:  (N,n_points=192,n_features=76)

    def recurrent(self, zt):
        '''
        GRU RNN
        '''
        batch_size = self.batch_size
        # output shape: (N,  n_frames, features,1)
        hidden = self.init_hidden(batch_size)
        hidden = torch.cat((hidden, hidden), dim=0)
        hidden = (hidden, hidden)
        output, hidden = self.gru(zt, hidden)
        return output, hidden

    def gru_to_ct(self, zt):
        '''
        return the last time_step of GRU result
        '''
        output, hidden = self.recurrent(zt)
        c_t = output[:, -1, :].view(self.batch_size, self.gru_out)
        return c_t, hidden

    def compute_nce(self, encode_samples, pred):
        '''
        -----------------------------------------------------------------------------------
                        --------------Calculate NCE loss--------------
        -----------------------------------------------------------------------------------
        ...argument:
        ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
        ......pred :            Wk[i]( C_t )
        '''
        nce = 0  # average over time_step and batch
        for i in np.arange(0, self.time_step):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            # print(total)
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.batch_size * self.time_step
        accuracy = 1. * correct.item() / self.batch_size

        return nce, accuracy

    def get_reg_out(self, x):
        self.batch_size = x.shape[0]

        x = x.squeeze(1).transpose(1, 2)
        self.n_frames = x.shape[2]

        z = self.encode(x).transpose(1, 2)
        z = self.linear(z)

        forward_seq = z[:, :, :]
        c_t, hidden = self.gru_to_ct(forward_seq)
        return c_t

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.encode(x).transpose(1, 2)  # z: (batch, n_time, conv[-1])
        d = self.decode(z.transpose(1, 2))

        self.batch_size = x.shape[0]
        self.n_frames = x.shape[2]

        # make change to here
        # t_samples should at least start from 30
        t_samples = torch.randint(low=self.time_step, high=self.n_frames - self.time_step - 1, size=(1,)).long().to(
            device)
        #
        encode_samples = torch.empty((self.time_step, self.batch_size, self.embedded_features)).float().to(
            device)  # e.g.
        # size

        z = self.linear(z)
        for i in np.arange(1, self.time_step + 1):
            encode_samples[i - 1, :, :] = z[:, int(t_samples) + i, :]

        forward_seq = z[:, :int(t_samples) + 1, :]

        c_t, hidden = self.gru_to_ct(forward_seq)

        pred = torch.empty((self.time_step, self.batch_size, self.embedded_features)).float().to(device)
        for i in np.arange(0, self.time_step):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        # d = self.decode(pred.transpose(1,2).transpose(0,2))
        nce, accuracy = self.compute_nce(encode_samples, pred)

        return d, nce, accuracy


class CPAE2_S(CPAE1_S):
    """
    Use conv1dtranspose in CPAE1
    """

    def __init__(
            self,
            embedded_features,
            gru_out,
            conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8],
            time_step=30,
            n_points=192,
            n_features=76,
    ):
        self.embedded_features = embedded_features
        self.gru_out = gru_out
        self.conv_sizes = conv_sizes
        self.time_step = time_step
        # kernel_sizes=get_kernel_sizes() #TODO
        super(CPAE2_S, self).__init__()

        # . If is int, uses the same padding in all boundaries.
        #  If a 4-tuple, uses (left ,right ,top ,bottom )
        self.channels = [n_features] + conv_sizes

        # the core part of model list
        self.enSequential = lambda inChannel, outChannel: nn.Sequential(
            nn.ReflectionPad1d((0, 1)),
            nn.Conv1d(inChannel, outChannel, kernel_size=2, padding=0),
            nn.BatchNorm1d(outChannel),
            nn.ReLU(inplace=True)
        )
        self.deSequential = lambda inChannel, outChannel: nn.Sequential(
            nn.ConvTranspose1d(inChannel, outChannel, kernel_size=3, padding=1),
            nn.BatchNorm1d(outChannel),
            nn.ReLU(inplace=True)
        )

        # ** minded the length should be 1 element shorter than # of channels
        self.encoder = nn.ModuleList(
            [self.enSequential(self.channels[i], self.channels[i + 1]) for i in range(len(conv_sizes))]
        )

        self.decode_channels = self.channels[::-1]
        self.decoder = nn.ModuleList(
            [self.deSequential(self.decode_channels[i], self.decode_channels[i + 1]) for i in range(len(conv_sizes))]
        )
        self.linear = nn.Linear(self.conv_sizes[-1], self.embedded_features).to(device)

        self.Wk = nn.ModuleList([nn.Linear(self.gru_out, self.embedded_features) for i in range(self.time_step)]).to(
            device)

        # dim = 1 !!!
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.gru = nn.GRU(
            self.embedded_features,
            gru_out,
            num_layers=1,
            bidirectional=False,
            batch_first=True).to(device)
        self.beforeNCE = None

        # input shape: (N,C=1,n_points=192,n_features=76)
        # output shape: (N, C=sizes[-1], )
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(self._weights_init)
        # def relevant_points(n):
        # deconvolution     nn. unMaxPool


class CPAE3_S(CPAE2_S):
    """
    Use conv1dtranspose in CPAE1  & Maxpooling & unpooing
    """

    def __init__(
            self,
            embedded_features,
            gru_out,
            conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8],
            time_step=30,
            n_points=192,
            n_features=76,
    ):
        self.embedded_features = embedded_features
        self.gru_out = gru_out
        self.conv_sizes = conv_sizes
        self.time_step = time_step
        # kernel_sizes=get_kernel_sizes() #TODO
        super(CPAE3_S, self).__init__()
        self.n_features = n_features
        # . If is int, uses the same padding in all boundaries.
        #  If a 4-tuple, uses (left ,right ,top ,bottom )
        self.channels = [n_features] + conv_sizes
        self.decode_channels = self.channels[::-1]

        encodelist = []
        count = 0
        for i, j in zip(self.channels[:-1], self.channels[1:]):
            encodelist.append(nn.ReflectionPad1d((0, 1)))
            encodelist.append(nn.Conv1d(i, j, kernel_size=2, padding=0))
            encodelist.append(nn.BatchNorm1d(j))
            encodelist.append(nn.ReLU(inplace=True))
            if count < 2:
                encodelist.append(nn.ReflectionPad1d((0, 1)))
                encodelist.append(nn.MaxPool1d(2, stride=1))
            count += 1
        self.encoder = nn.Sequential(*encodelist)
        decodelist = []
        count = 0
        for i, j in zip(self.decode_channels[:-1], self.decode_channels[1:]):
            decodelist.append(nn.ConvTranspose1d(i, j, kernel_size=3, padding=1))
            decodelist.append(nn.BatchNorm1d(j))
            decodelist.append(nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(*decodelist)

        self.linear = nn.Linear(self.conv_sizes[-1], self.embedded_features).to(device)

        self.Wk = nn.ModuleList([nn.Linear(self.gru_out, self.embedded_features) for i in range(self.time_step)]).to(
            device)

        # dim = 1 !!!
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.gru = nn.GRU(
            self.embedded_features,
            gru_out,
            num_layers=1,
            bidirectional=False,
            batch_first=True).to(device)
        self.beforeNCE = None

        # input shape: (N,C=1,n_points=192,n_features=76)
        # output shape: (N, C=sizes[-1], )
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(self._weights_init)
        # def relevant_points(n):
        # deconvolution     nn. unMaxPool


class CPAE4_S(CPAE1_S):
    def __int__(self):
        super(CPAE4_S, self).__init__()

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        self.batch_size = x.shape[0]
        self.n_frames = x.shape[2]
        x = x.transpose(1, 2)
        z = self.encode(x).transpose(1, 2)  # z: (batch, n_time, conv[-1])
        z = self.linear(z)
        x = x.transpose(1, 2)

        # make change to here
        # t_samples should at least start from 30
        t_samples = torch.randint(low=self.time_step, high=self.n_frames - self.time_step - 1, size=(1,)).long().to(
            device)

        forward_seq = z[:, :int(t_samples) + 1, :]

        c_t, hidden = self.gru_to_ct(forward_seq)

        pred = torch.empty((self.time_step, self.batch_size, self.embedded_features)).float().to(device)

        for i in np.arange(0, self.time_step):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        #
        x_samples = torch.empty((self.time_step, self.batch_size, self.n_features)).float().to(
            device)  # e.g.
        # size

        for i in np.arange(1, self.time_step + 1):
            x_samples[i - 1, :, :] = x[:, int(t_samples) + i, :]

        reconstruct_samples = self.decode(pred.transpose(1, 2)).transpose(1, 2)

        # d = self.decode(pred.transpose(1,2).transpose(0,2))
        nce, accuracy = self.compute_nce(x_samples, reconstruct_samples)

        return accuracy, nce, x

    def compute_nce(self, encode_samples, pred):
        '''
        -----------------------------------------------------------------------------------
                        --------------Calculate NCE loss--------------
        -----------------------------------------------------------------------------------
        ...argument:
        ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
        ......pred :            Wk[i]( C_t )
        '''
        nce = 0  # average over time_step and batch
        for i in np.arange(0, self.time_step):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            # print(total)
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.batch_size * self.time_step
        accuracy = 1. * correct.item() / self.batch_size

        return nce, accuracy


class CPAE4_NO_BN(CPAE1_NO_BN):
    def __int__(self):
        super(CPAE4_NO_BN, self).__init__()

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        self.batch_size = x.shape[0]
        self.n_frames = x.shape[2]
        x = x.transpose(1, 2)
        z = self.encode(x).transpose(1, 2)  # z: (batch, n_time, conv[-1])
        z = self.linear(z)
        x = x.transpose(1, 2)

        # make change to here
        # t_samples should at least start from 30
        t_samples = torch.randint(low=self.time_step, high=self.n_frames - self.time_step - 1, size=(1,)).long().to(
            device)
        forward_seq = z[:, :int(t_samples) + 1, :]

        c_t, hidden = self.gru_to_ct(forward_seq)

        pred = torch.empty((self.time_step, self.batch_size, self.embedded_features)).float().to(device)

        for i in np.arange(0, self.time_step):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        #
        x_samples = torch.empty((self.time_step, self.batch_size, self.n_features)).float().to(
            device)  # e.g.
        # size

        for i in np.arange(1, self.time_step + 1):
            x_samples[i - 1, :, :] = x[:, int(t_samples) + i, :]

        reconstruct_samples = self.decode(pred.transpose(1, 2)).transpose(1, 2)

        # d = self.decode(pred.transpose(1,2).transpose(0,2))
        nce, accuracy = self.compute_nce(x_samples, reconstruct_samples)

        return accuracy, nce, x

    def compute_nce(self, encode_samples, pred):
        '''
        -----------------------------------------------------------------------------------
                        --------------Calculate NCE loss--------------
        -----------------------------------------------------------------------------------
        ...argument:
        ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
        ......pred :            Wk[i]( C_t )
        '''
        nce = 0  # average over time_step and batch
        for i in np.arange(0, self.time_step):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            # print(total)
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.batch_size * self.time_step
        accuracy = 1. * correct.item() / self.batch_size

        return nce, accuracy


class CPAE7_S(CPAE4_S):
    """
    this CPAE simply make `f_i(x_i,x_j)` the chimera_loss function
    """

    def __init__(self, embedded_featrues=8, gru_out=8, Lambda=[1, 1, 3]):
        super(CPAE7_S, self).__init__(embedded_featrues,
                                      gru_out)  # to initiate the CPAE4 with embedded_featrues = 8, gru_out = 8

        self.Lambda = torch.tensor(Lambda).float().cuda()
        self.Lambda = self.Lambda / sum(self.Lambda) * 10

    def weighted_mask(self, x):
        """
        similar to chimera loss
        """
        #         x = x.transpose(0,1)
        #         d = d.transpose(0,1)
        assert (x.shape[1] == 76)

        mse_m = torch.ones(x.shape).to(device)
        mask_m, mapping_m = mask_mapping_M(x)

        return self.Lambda[0] * mse_m + self.Lambda[1] * mask_m + self.Lambda[2] * mapping_m

    def compute_nce(self, x, d):
        '''
        -----------------------------------------------------------------------------------
                        --------------Calculate NCE loss--------------
        -----------------------------------------------------------------------------------
        ...argument:
        ......x : x_samples , ( time_step, batch_size, conv_sizes[-1] )
        ......d : reconstruct_samples , the same shape as x,  self.decode(z_hat)
        '''
        nce = 0  # average over time_step and batch
        for i in np.arange(0, self.time_step):
            x_w = self.weighted_mask(x[i]) * x[i]
            total = torch.mm(x_w, torch.transpose(d[i], 0, 1))  # e.g. size 8*8
            # print(total)
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.batch_size * self.time_step
        accuracy = 1. * correct.item() / self.batch_size

        return nce, accuracy


class CPLSTM(nn.Module):
    """
    Bi-directional LSTM
    """

    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5):
        # Smart way to filter the args

        self.dim = dim
        self.bn = bn
        self.drop = dropout
        self.task = task
        self.depth = depth
        self.time_step = time_step
        self.num_classes = num_classes
        self.input_dim = input_dim
        super(CPLSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=dim // 2,
            dropout=self.drop,
            bidirectional=True,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=dim,
            dropout=self.drop,
            bidirectional=False,
            batch_first=True
        )

        self.Wk = nn.ModuleList([nn.Linear(self.dim, self.dim) for i in range(self.time_step)])
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)

    def encodeRegress(self, x):
        x, _ = self.lstm1(x)
        x, state = self.lstm2(x)
        ht, ct = state
        return x, ht, ct

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        self.bs = x.shape[0]
        c_preds = [0] * self.time_step
        c_latent = []
        xt, ht, ct = self.encodeRegress(x[:, :t, :])
        h, c = ht, ct
        for i in range(1, self.time_step + 1):
            c_preds[i - 1] = self.Wk[i - 1](ht)
            _, h, c = self.encodeRegress(x[:, t + i, :])
            c_latent.append(c)

        nce = 0
        for i in np.arange(0, self.time_step):
            total = torch.mm(c_latent[i].squeeze(0), torch.transpose(c_preds[i].squeeze(0), 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.bs).to(device)))

            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.bs * self.time_step
        accuracy = 1. * correct.item() / self.bs

        return accuracy, nce, c

    def get_reg_out(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        # print('reshape x to ',x.shape)
        xt, ht, ct = self.encodeRegress(x[:, :, :])
        # print(ht.shape)
        return xt.reshape((x.shape[0], -1))


class CPLSTM2(nn.Module):
    """
    LSTM
    """

    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5):
        # Smart way to filter the args

        self.dim = dim
        self.bn = bn
        self.drop = dropout
        self.task = task
        self.depth = depth
        self.time_step = time_step
        self.num_classes = num_classes
        self.input_dim = input_dim
        super(CPLSTM2, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=dim,
            dropout=self.drop,
            bidirectional=False,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=dim,
            dropout=self.drop,
            bidirectional=False,
            batch_first=True
        )

        self.Wk = nn.ModuleList([nn.Linear(self.dim, self.dim) for i in range(self.time_step)])
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)

    def encodeRegress(self, x):
        x, _ = self.lstm1(x)
        x, state = self.lstm2(x)
        ht, ct = state
        return x, ht, ct

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        self.bs = x.shape[0]
        c_preds = [0] * self.time_step
        c_latent = []
        xt, ht, ct = self.encodeRegress(x[:, :t + 1, :])
        h, c = ht, ct
        for i in range(1, self.time_step + 1):
            c_preds[i - 1] = self.Wk[i - 1](ht)
            _, h, c = self.encodeRegress(x[:, t + i, :])
            c_latent.append(c)

        nce = 0
        for i in np.arange(0, self.time_step):
            total = torch.mm(c_latent[i].squeeze(0), torch.transpose(c_preds[i].squeeze(0), 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.bs).to(device)))

            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.bs * self.time_step
        accuracy = 1. * correct.item() / self.bs

        return accuracy, nce, c

    def get_reg_out(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        # print('reshape x to ',x.shape)
        xt, ht, ct = self.encodeRegress(x[:, :, :])
        print(xt.shape)
        return xt
        # return xt.reshape((x.shape[0],-1))


class CPLSTM3(nn.Module):
    """
    CPLSTM2 with dropout in non-recurrent layers and FC added.
    """

    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5):
        # Smart way to filter the args

        self.dim = dim
        self.bn = bn
        self.drop = dropout
        self.task = task
        self.depth = depth
        self.time_step = time_step
        self.num_classes = num_classes
        self.input_dim = input_dim
        super(CPLSTM3, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )

        self.dropout = nn.Dropout(self.drop)
        self.Wk = nn.ModuleList([nn.Linear(self.dim, self.dim) for i in range(self.time_step)])
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.fcs = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, self.dim)
        )
        for model in [self.lstm1, self.lstm2, self.fcs]:
            self.initialize_weights(model)
        for model in self.Wk:
            self.initialize_weights(model)

    def init_hidden(self, bs, dim):
        cell_states = torch.zeros(1, bs, dim).to(device)
        hidden_states = torch.zeros(1, bs, dim).to(device)
        return (hidden_states, cell_states)

    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

    def encodeRegress(self, x, warm=False):
        bs = x.shape[0]
        x = self.dropout(x)
        (h0, c0) = self.init_hidden(bs, self.dim)
        if warm:
            x_temp, state1 = self.lstm1(x[:, :5, :], (h0, c0))
            _, state2 = self.lstm2(x_temp[:, :5, :], (h0, c0))
            # print([i.shape for i in state1],h0.shape,c0.shape)
            x, state1 = self.lstm1(x[:, :, :], state1)
            x, state2 = self.lstm2(x[:, :, :], state2)
            ht, ct = state2
        else:
            x, state1 = self.lstm1(x[:, :, :], (h0, c0))
            x, state2 = self.lstm2(x[:, :, :], (h0, c0))
            ht, ct = state2
        return x, ht, ct

    #
    #
    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        self.bs = x.shape[0]
        c_preds = [0] * self.time_step
        c_latent = []
        xt, ht, ct = self.encodeRegress(x[:, :t, :])
        h, c = ht, ct
        for i in range(1, self.time_step + 1):
            c_preds[i - 1] = self.fcs(self.Wk[i - 1](ht))
            _, h, c = self.encodeRegress(x[:, t + i, :])
            c_latent.append(self.fcs(c))

        nce = 0
        for i in np.arange(0, self.time_step):
            total = torch.mm(c_latent[i].squeeze(0), torch.transpose(c_preds[i].squeeze(0), 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.bs).to(device)))

            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.bs * self.time_step
        accuracy = 1. * correct.item() / self.bs

        return accuracy, nce, c

    def get_reg_out(self, x, stack=False, warm=False, conti=False):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        # print('reshape x to ',x.shape)
        xt, ht, ct = self.encodeRegress(x[:, :, :], warm)
        # print(ht.shape)
        # return xt.reshape((x.shape[0],-1))
        if stack: return torch.cat((xt.reshape((x.shape[0], -1)), ct.squeeze(0)), 1)
        return xt[:, -1, :].squeeze(1)


class CPLSTM4(nn.Module):
    """
    CPLSTM4------use lstm as Wk
    mode=1 use hidden states when predict. else use cell states
    """

    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5, mode=1, noct=False, switch=True):
        self.dim = dim
        self.bn = bn
        self.drop = dropout
        self.task = task
        self.depth = depth
        self.time_step = time_step
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.mode = mode
        self.noct = noct
        super(CPLSTM4, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )
        self.lstm3 = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )
        if self.noct:
            self.stack_dim = self.dim * 192
        else:
            self.stack_dim = self.dim * 193
        self.dropout = nn.Dropout(self.drop)
        # self.Wk = nn.ModuleList([nn.Linear(self.dim, self.dim) for i in range(self.time_step)])
        self.switch = switch
        if self.switch == False:
            self.softmax = nn.Softmax(dim=1)
            self.lsoftmax = nn.LogSoftmax(dim=1)
        else:
            self.softmax = nn.Softmax(dim=0)
            self.lsoftmax = nn.LogSoftmax(dim=0)

        self.fcs = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.dim, self.dim)
        )
        for model in [self.lstm1, self.lstm2, self.lstm3, self.fcs]:
            self.initialize_weights(model)

    def init_hidden(self, bs, dim):
        cell_states = torch.zeros(1, bs, dim).to(device)
        hidden_states = torch.zeros(1, bs, dim).to(device)
        return (hidden_states, cell_states)

    def freeze_encode(self):
        for param in self.lstm1.parameters():
            param.requires_grad = False

    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

    def encodeRegress(self, x, warm=False, conti=False):
        bs = x.shape[0]
        x = self.dropout(x)
        if conti:
            x, state1 = self.lstm1(x)
            x, state2 = self.lstm2(x)
            ht, ct = state2
            return x, ht, ct

        (h0, c0) = self.init_hidden(bs, self.dim)
        if warm:
            x_temp, state1 = self.lstm1(x[:, :5, :], (h0, c0))
            _, state2 = self.lstm2(x_temp[:, :5, :], (h0, c0))
            # print([i.shape for i in state1],h0.shape,c0.shape)
            x, state1 = self.lstm1(x[:, :, :], state1)
            x, state2 = self.lstm2(x[:, :, :], state2)
            ht, ct = state2
        else:
            x, state1 = self.lstm1(x[:, :, :], (h0, c0))
            x, state2 = self.lstm2(x[:, :, :], (h0, c0))
            ht, ct = state2
        return x, ht, ct

    #
    def predict(self, z, hz, cz, ts, mode=1):
        """"
        if mode==1: return hidden states; else return cell states"""
        h, c = hz, cz
        x_previous = z
        c_preds = torch.empty((self.time_step, self.bs, self.dim)).to(device)
        for i in range(ts):
            x_pred, (h, c) = self.lstm3(x_previous, (h, c))
            if mode:
                c_preds[i, :, :] = h
            else:
                c_preds[i, :, :] = c  # mode = 0
            x_previous = x_pred
        return c_preds

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        self.bs = x.shape[0]
        c_latent = []
        xt, ht, ct = self.encodeRegress(x[:, :t + 1, :])
        c_preds = self.fcs(self.predict(ht.transpose(0, 1), ht, ct, self.time_step, self.mode))

        for i in range(1, self.time_step + 1):
            _, h, c = self.encodeRegress(x[:, t + i, :])  # init with zeros
            c_latent.append(self.fcs(c))
        nce = 0
        for i in np.arange(0, self.time_step):
            total = torch.mm(c_latent[i].squeeze(0), torch.transpose(c_preds[i].squeeze(0), 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.bs).to(device)))

            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.bs * self.time_step
        accuracy = 1. * correct.item() / self.bs

        return accuracy, nce, c

    def get_reg_out(self, x, stack=False, warm=False, conti=False):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        # print('reshape x to ',x.shape)
        xt, ht, ct = self.encodeRegress(x[:, :, :], warm, conti)
        # print(ht.shape)
        # return xt.reshape((x.shape[0],-1))
        if stack and self.noct: return self.dropout(xt.reshape((x.shape[0], -1)))

        if stack: return self.dropout(torch.cat((xt.reshape((x.shape[0], -1)), ct.squeeze(0)), 1))
        return xt[:, -1, :].squeeze(1)


class CPLSTM4C(nn.Module):
    """
    re-init hidden at time point t
    mode=1 use hidden states when predict. else use cell states
    """

    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5, mode=1, noct=False):
        self.dim = dim
        self.bn = bn
        self.drop = dropout
        self.task = task
        self.depth = depth
        self.time_step = time_step
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.mode = mode
        self.noct = noct
        super(CPLSTM4C, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )
        self.lstm3 = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )
        if self.noct:
            self.stack_dim = self.dim * 192
        else:
            self.stack_dim = self.dim * 193
        self.dropout = nn.Dropout(self.drop)
        # self.Wk = nn.ModuleList([nn.Linear(self.dim, self.dim) for i in range(self.time_step)])
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.fcs = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.dim, self.dim)
        )
        for model in [self.lstm1, self.lstm2, self.lstm3, self.fcs]:
            self.initialize_weights(model)

    def init_hidden(self, bs, dim):
        cell_states = torch.zeros(1, bs, dim).to(device)
        hidden_states = torch.zeros(1, bs, dim).to(device)
        return (hidden_states, cell_states)

    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

    def encodeRegress(self, x, warm=False, conti=False):
        bs = x.shape[0]
        x = self.dropout(x)
        if conti:
            x, state1 = self.lstm1(x)
            x, state2 = self.lstm2(x)
            ht, ct = state2
            return x, ht, ct

        (h0, c0) = self.init_hidden(bs, self.dim)
        if warm:
            x_temp, state1 = self.lstm1(x[:, :5, :], (h0, c0))
            _, state2 = self.lstm2(x_temp[:, :5, :], (h0, c0))
            # print([i.shape for i in state1],h0.shape,c0.shape)
            x, state1 = self.lstm1(x[:, :, :], state1)
            x, state2 = self.lstm2(x[:, :, :], state2)
            ht, ct = state2
        else:
            x, state1 = self.lstm1(x[:, :, :], (h0, c0))
            x, state2 = self.lstm2(x[:, :, :], (h0, c0))
            ht, ct = state2
        return x, ht, ct

    #
    def predict(self, z, hz, cz, ts, mode=1):
        """"
        if mode==1: return hidden states; else return cell states"""
        h, c = hz, cz
        x_previous = z
        c_preds = torch.empty((self.time_step, self.bs, self.dim)).to(device)
        for i in range(ts):
            x_pred, (h, c) = self.lstm3(x_previous, (h, c))
            if mode:
                c_preds[i, :, :] = h
            else:
                c_preds[i, :, :] = c
            x_previous = x_pred
        return c_preds

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        self.bs = x.shape[0]
        c_latent = []
        xt, ht, ct = self.encodeRegress(x[:, :t + 1, :])
        c_preds = self.fcs(self.predict(ht.transpose(0, 1), ht, ct, self.time_step, self.mode))

        (h0, c0) = self.init_hidden(self.bs, self.dim)
        h1, c1 = h0, c0
        h2, c2 = h0, c0
        for i in range(1, self.time_step + 1):
            # BUG  : self.time_step ? i
            tmp, (h1, c1) = self.lstm1(x[:, t + i, :], (h1, c1))
            _, (h2, c2) = self.lstm2(tmp, (h2, c2))
            c_latent.append(self.fcs(c2))
        nce = 0
        for i in np.arange(0, self.time_step):
            total = torch.mm(c_latent[i].squeeze(0), torch.transpose(c_preds[i].squeeze(0), 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.bs).to(device)))

            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.bs * self.time_step
        accuracy = 1. * correct.item() / self.bs

        return accuracy, nce, None

    def get_reg_out(self, x, stack=False, warm=False, conti=False):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        # print('reshape x to ',x.shape)
        xt, ht, ct = self.encodeRegress(x[:, :, :], warm, conti)
        # print(ht.shape)
        # return xt.reshape((x.shape[0],-1))
        if stack and self.noct: return self.dropout(xt.reshape((x.shape[0], -1)))

        if stack: return self.dropout(torch.cat((xt.reshape((x.shape[0], -1)), ct.squeeze(0)), 1))
        return xt[:, -1, :].squeeze(1)


class CPLSTM3H(CPLSTM3):
    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5):
        super(CPLSTM3H, self).__init__(dim, bn, dropout, task,
                                       depth, num_classes,
                                       input_dim, time_step)

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        self.bs = x.shape[0]
        c_preds = [0] * self.time_step
        c_latent = []
        xt, ht, ct = self.encodeRegress(x[:, :t, :])
        h, c = ht, ct
        for i in range(1, self.time_step + 1):
            c_preds[i - 1] = self.fcs(self.Wk[i - 1](ht))
            _, h, c = self.encodeRegress(x[:, t + i, :])
            c_latent.append(self.fcs(h))

        nce = 0
        for i in np.arange(0, self.time_step):
            total = torch.mm(c_latent[i].squeeze(0), torch.transpose(c_preds[i].squeeze(0), 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.bs).to(device)))

            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.bs * self.time_step
        accuracy = 1. * correct.item() / self.bs

        return accuracy, nce, c


class CPLSTM4H(CPLSTM4):
    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5, mode=1):
        super(CPLSTM4H, self).__init__(dim, bn, dropout, task,
                                       depth, num_classes,
                                       input_dim, time_step, mode)

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        self.bs = x.shape[0]
        # xt, ht, ct = self.encodeRegress(x[:, :t + 1, :])
        # c_preds = self.fcs(self.predict(ht.transpose(0, 1), ht, ct, self.time_step, self.mode))
        #
        # # for i in range(1, self.time_step + 1):
        # x, (h, c) = self.lstm1(x[:, t + 1:t+self.time_step+1, :])
        # c_latent=self.fcs(x)

        z_embeds, _ = self.lstm1(x)
        _, (hidden_ct, cell_ct) = self.lstm2(z_embeds[:, :t + 1, :])

        z_preds_time_step = self.fcs(
            self.predict(hidden_ct.transpose(0, 1), hidden_ct, cell_ct, self.time_step, self.mode))
        z_embeds_time_step = z_embeds[:, t + 1:t + self.time_step + 1, :]

        nce = 0
        for i in np.arange(0, self.time_step):
            total = torch.mm(z_embeds_time_step[:, i, :].squeeze(0),
                             torch.transpose(z_preds_time_step[i].squeeze(0), 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.bs).to(device)))

            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.bs * self.time_step
        accuracy = 1. * correct.item() / self.bs

        return accuracy, nce, None


class CPAELSTM41(nn.Module):
    """
    CPLSTM4------use lstm as Wk
    mode=1 use hidden states when predict. else use cell states
    """

    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5, mode=1, noct=False):
        self.dim = dim
        self.bn = bn
        self.drop = dropout
        self.task = task
        self.depth = depth
        self.time_step = time_step
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.mode = mode
        self.noct = noct
        super(CPAELSTM41, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )
        self.lstm3 = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True)

        self.dropout = nn.Dropout(self.drop)
        # self.Wk = nn.ModuleList([nn.Linear(self.dim, self.dim) for i in range(self.time_step)])
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.de_fc = nn.Sequential(nn.Linear(self.dim, self.input_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.input_dim, self.input_dim),
                                   nn.ReLU(inplace=True),
                                   )
        self.fcs = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.input_dim, self.input_dim)
        )
        for model in [self.lstm1, self.lstm2, self.lstm3, self.fcs]:
            self.initialize_weights(model)

    def init_hidden(self, bs, dim):
        cell_states = torch.zeros(1, bs, dim).to(device)
        hidden_states = torch.zeros(1, bs, dim).to(device)
        return (hidden_states, cell_states)

    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

    def encodeRegress(self, x, warm=False, conti=False):
        bs = x.shape[0]
        x = self.dropout(x)
        if conti:
            x, state1 = self.lstm1(x)
            x, state2 = self.lstm2(x)
            ht, ct = state2
            return x, ht, ct

        (h0, c0) = self.init_hidden(bs, self.dim)
        if warm:
            x_temp, state1 = self.lstm1(x[:, :5, :], (h0, c0))
            _, state2 = self.lstm2(x_temp[:, :5, :], (h0, c0))
            # print([i.shape for i in state1],h0.shape,c0.shape)
            x, state1 = self.lstm1(x[:, :, :], state1)
            x, state2 = self.lstm2(x[:, :, :], state2)
            ht, ct = state2
        else:
            x, state1 = self.lstm1(x[:, :, :], (h0, c0))
            x, state2 = self.lstm2(x[:, :, :], (h0, c0))
            ht, ct = state2
        return x, ht, ct

    #
    #
    def predict(self, z, hz, cz, ts, mode=1):
        """"
        if mode==1: return hidden states; else return cell states"""
        h, c = hz, cz
        x_previous = z
        c_preds = torch.empty((self.time_step, self.bs, self.dim)).to(device)
        for i in range(ts):
            x_pred, (h, c) = self.lstm3(x_previous, (h, c))
            if mode:
                c_preds[i, :, :] = h
            else:
                c_preds[i, :, :] = c
            x_previous = x_pred
        return c_preds

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        self.bs = x.shape[0]
        c_latent = []
        xt, ht, ct = self.encodeRegress(x[:, :t + 1, :])
        x_preds = self.fcs(self.de_fc(self.predict(ht.transpose(0, 1), ht, ct, self.time_step, self.mode)))

        nce = 0
        for i in np.arange(0, self.time_step):
            total = torch.mm(self.fcs(x[:, t + i + 1, :]).squeeze(1), torch.transpose(x_preds[i].squeeze(0), 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.bs).to(device)))

            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.bs * self.time_step
        accuracy = 1. * correct.item() / self.bs

        return accuracy, nce, None

    def get_reg_out(self, x, stack=False, warm=False, conti=False):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        # print('reshape x to ',x.shape)
        xt, ht, ct = self.encodeRegress(x[:, :, :], warm, conti)
        # print(ht.shape)
        # return xt.reshape((x.shape[0],-1))
        if stack and self.noct: return self.dropout(xt.reshape((x.shape[0], -1)))

        if stack: return self.dropout(torch.cat((xt.reshape((x.shape[0], -1)), ct.squeeze(0)), 1))
        return xt[:, -1, :].squeeze(1)


class CPAELSTM42(CPAELSTM41):
    """
    two layer lstm as decoder to reconstruct x.
    """

    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5, mode=1):
        super(CPAELSTM42, self).__init__(dim, bn, dropout, task,
                                         depth, num_classes,
                                         input_dim, time_step, mode)
        self.lstm3 = nn.LSTM(
            input_size=self.input_dim,
            num_layers=1,
            hidden_size=self.input_dim,
            bidirectional=False,
            batch_first=True)
        #
        # self.dropout=nn.Dropout(self.drop)
        # # self.Wk = nn.ModuleList([nn.Linear(self.dim, self.dim) for i in range(self.time_step)])
        # self.softmax = nn.Softmax(dim=0)
        # self.lsoftmax = nn.LogSoftmax(dim=0)
        self.de_fc = nn.Sequential(
            nn.Linear(self.dim, self.input_dim),
            nn.ReLU(inplace=True)
        )
        # self.fcs=nn.Sequential(
        #   nn.Linear(self.input_dim,self.input_dim),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(self.input_dim,self.input_dim)
        # )
        # for model in [self.lstm1,self.lstm2,self.lstm3,self.fcs]:
        #     self.initialize_weights(model)

    def init_hidden(self, bs, dim):
        cell_states = torch.zeros(1, bs, dim).to(device)
        hidden_states = torch.zeros(1, bs, dim).to(device)
        return (hidden_states, cell_states)

    # BUG

    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

    def encodeRegress(self, x):
        bs = x.shape[0]
        x = self.dropout(x)
        (h0, c0) = self.init_hidden(bs, self.dim)
        x, _ = self.lstm1(x, (h0, c0))
        x, state = self.lstm2(x, (h0, c0))
        ht, ct = state
        return x, ht, ct

    #
    #
    def predict(self, z, hz, cz, ts, mode=1):
        """"
        if mode==1: return hidden states; else return cell states"""
        h, c = self.de_fc(hz), self.de_fc(cz)
        x_previous = self.de_fc(z)
        x_preds = torch.empty((self.time_step, self.bs, self.input_dim)).to(device)
        for i in range(ts):
            x_pred, (h, c) = self.lstm3(x_previous, (h, c))
            if mode:
                x_preds[i, :, :] = h
            else:
                x_preds[i, :, :] = c
            x_previous = x_pred
        return x_preds

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        self.bs = x.shape[0]
        c_latent = []
        xt, ht, ct = self.encodeRegress(x[:, :t + 1, :])
        x_preds = self.fcs(self.predict(ht.transpose(0, 1), ht, ct, self.time_step, self.mode))

        # for i in range(1,self.time_step+1):
        #     _, h,c=self.encodeRegress(x[:,t+i,:])
        #     c_latent.append(self.fcs(c))
        nce = 0
        for i in np.arange(0, self.time_step):
            total = torch.mm(self.fcs(x[:, t + i + 1, :]).squeeze(1), torch.transpose(x_preds[i].squeeze(0), 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.bs).to(device)))

            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.bs * self.time_step
        accuracy = 1. * correct.item() / self.bs

        return accuracy, nce, None

    def get_reg_out(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        # print('reshape x to ',x.shape)
        xt, ht, ct = self.encodeRegress(x[:, :, :])
        # print(ht.shape)
        # return xt.reshape((x.shape[0],-1))
        return xt[:, -1, :].squeeze(1)


class CPAELSTM43(CPLSTM4H):
    """
    add decoder constraint in loss function
    """

    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5, mode=1):
        super(CPAELSTM43, self).__init__(dim, bn, dropout, task,
                                         depth, num_classes,
                                         input_dim, time_step, mode)

        self.lstm4 = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.input_dim,
            bidirectional=False,
            batch_first=True)


    def compute_nce(self, encode_samples, pred):
        '''
        -----------------------------------------------------------------------------------
                        --------------Calculate NCE loss--------------
        -----------------------------------------------------------------------------------
        ...argument:
        ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
        ......pred :            Wk[i]( C_t )
        '''
        nce = 0  # average over time_step and batch
        self.batch_size = self.bs
        for i in np.arange(0, self.time_step):
            try:
                total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            except IndexError:
                print('i is : %s,latent shape: %s, pred shape: %s ' % (i, encode_samples.shape, pred.shape))
                raise AssertionError
            # print(total)
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * self.batch_size * self.time_step
        accuracy = 1. * correct.item() / self.batch_size

        return nce, accuracy


    def encode(self, x):
        bs = x.shape[0]
        x = self.dropout(x)
        (h0, c0) = self.init_hidden(bs, self.dim)
        x, _ = self.lstm1(x, (h0, c0))
        return x


    def decode(self, x):
        bs = x.shape[0]
        x = self.dropout(x)
        (h0, c0) = self.init_hidden(bs, self.input_dim)
        x, _ = self.lstm4(x, (h0, c0))
        return x


    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        x_ori = x
        t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        self.bs = x.shape[0]
        c_latent = []
        xt, ht, ct = self.encodeRegress(x[:, :t + 1, :])
        c_preds = self.fcs(self.predict(ht.transpose(0, 1), ht, ct, self.time_step, self.mode))

        # for i in range(1, self.time_step + 1):
        # x, h, c = self.encodeRegress(x[:, t + 1:t+self.time_step+1, :])
        z, (h, c) = self.lstm1(x)
        c_latent = self.fcs(z[:, t + 1:t + self.time_step + 1, :])  # with memory
        x_hat = self.decode(z)
        nce, acc = self.compute_nce(c_latent.transpose(0, 1), c_preds)

        return x_hat, nce, acc


class CPAELSTM44(CPLSTM4):
    """
    add decoder constraint in loss function
    sim: similarity function. 'dot' for dot product, 'cosine' for cosine similarity
    """

    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5, t_range=None,mode=1,sym=False, sim='dot',temperature=1,pred_mode='step'):
        super(CPAELSTM44, self).__init__(dim, bn, dropout, task,
                                         depth, num_classes,
                                         input_dim, time_step, mode)

        self.lstm4 = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.input_dim,
            bidirectional=False,
            batch_first=True)
        self.sym=sym
        self.sim=sim 
        self.temperature=temperature
        self.t_range=t_range
        self.pred_mode = pred_mode
        if self.pred_mode=='future':
            self.W_pred = nn.Linear(self.dim, self.dim)

    def sim_func(self,a,b):
        if self.sim=='cosine':
            print('use cosine')
            a=a/a.norm(dim=-1,keepdim=True)
            b=b/b.norm(dim=-1,keepdim=True)
            a=self.temperature*a
            b=self.temperature*b
            return torch.mm(a,b.T)
        elif self.sim=='dot':
            print('use dot')
            return torch.mm(a,b.T)
            

    def compute_nce(self, encode_samples, pred):
        '''
        -----------------------------------------------------------------------------------
                        --------------Calculate NCE loss--------------
        -----------------------------------------------------------------------------------
        ...argument:
        ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
        ......pred :            Wk[i]( C_t )
        '''
        nce = 0  # average over time_step and batch
        self.batch_size = self.bs
        for i in np.arange(0, self.time_step):
            try:
                total = self.sim_func(encode_samples[i], pred[i])  # e.g. size 8*8
            except IndexError:
                print('i is : %s,latent shape: %s, pred shape: %s ' % (i, encode_samples.shape, pred.shape))
                raise AssertionError
            # print(total)
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            if self.sym:
                nce += 1/2*(torch.sum(torch.diag((nn.LogSoftmax(dim=0)(total)))) + torch.sum(torch.diag((nn.LogSoftmax(dim=1)(total)))))# nce is a tensor

            else:
                nce += torch.sum(torch.diag(self.lsoftmax(total))) 
        nce /= -1. * self.batch_size * self.time_step
        accuracy = 1. * correct.item() / self.batch_size

        return nce, accuracy

    def encode(self, x):
        bs = x.shape[0]
        x = self.dropout(x)
        (h0, c0) = self.init_hidden(bs, self.dim)
        x, _ = self.lstm1(x, (h0, c0))
        return x

    def decode(self, x):
        bs = x.shape[0]
        x = self.dropout(x)
        (h0, c0) = self.init_hidden(bs, self.input_dim)
        x, _ = self.lstm4(x, (h0, c0))
        return x

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        x_ori = x
        t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        self.bs = x.shape[0]
        c_latent = []
        xt, ht, ct = self.encodeRegress(x[:, :t + 1, :])
        c_preds = self.fcs(self.predict(ht.transpose(0, 1), ht, ct, self.time_step, self.mode))

        # for i in range(1, self.time_step + 1):
        # x, h, c = self.encodeRegress(x[:, t + 1:t+self.time_step+1, :])
        z_after_t = []

        for i in range(1, self.time_step + 1):
            _, h, c = self.encodeRegress(x[:, t + i, :])
            z_after_t.append(self.fcs(c))
        z_after_t = torch.cat(z_after_t, 0)

        c_embeds = self.fcs(z_after_t)
        z_all = torch.cat((xt, z_after_t.transpose(0, 1)), 1)
        x_hat = self.decode(z_all)
        nce, acc = self.compute_nce(c_embeds, c_preds)

        return x_hat, nce, acc

        def pred_future(self, x):
            x=self.check_input(x)

            # print(self.t_range)
            # print(self.max_len)
            t_range=(self.max_len*self.t_range[0],self.max_len*self.t_range[1])
            # print(t_range)
            # t_range = (self.max_len *2// 3, 4 * self.max_len // 5)
            # print(x.shape)
            x_ori = x
            if self.max_len>192: t=192
            else:
                t = torch.randint(low=int(t_range[0]), high=int(t_range[1]), size=(1,)).long()  # choose a point to split the time series
            # print('t is %s'%t)
            # self.bs = x.shape[0]
            latent_past, _, hidden_reg_out_past, _ = self.encodeRegress(x[:, :t + 1, :])
            latent_future, _, hidden_reg_out_future, _ = self.encodeRegress(x[:, t + 1:self.max_len, :])
            del x
            hidden_reg_out_pred = self.fcs(self.W_pred(hidden_reg_out_past))

            latent_all = torch.cat((latent_past, latent_future), 1)
            del latent_future,latent_past
            latent_all_attention = torch.mul(latent_all, self.cal_att2(latent_all))
            del latent_all
            x_hat = self.decode(latent_all_attention)
            nce, acc = self.compute_nce(self.fcs(hidden_reg_out_future), hidden_reg_out_pred)
            return x_hat, nce, acc



class CPAELSTM44_AT(CPLSTM4):
    """
    add decoder constraint in loss function
    pred_mode: 'step' for timestep prediction
                'future' for using past to predict future
    """

    def __init__(self, dim, bn, dropout, task,t_range=None,
                 depth=2, num_classes=1,
                 input_dim=76, flat_attention=False,time_step=5, sim='dot',temperature=1,mode=1, switch=True, pred_mode='step',sym=False):

        super(CPAELSTM44_AT, self).__init__(dim, bn, dropout, task,
                                            depth, num_classes,
                                            input_dim, time_step, mode, switch)

        self.lstm4 = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.input_dim,
            bidirectional=False,
            batch_first=True)
        self.att1 = nn.Linear(self.dim, self.dim)
        self.att2 = nn.Linear(self.dim, self.dim)
        self.flat_attention=flat_attention
        self.sim=sim 
        self.temperature=temperature
        self.t_range=t_range
        self.pred_mode = pred_mode
        if self.pred_mode=='future':
            self.W_pred = nn.Linear(self.dim, self.dim)
        self.sym=sym #whether use symmetric loss

    def cal_att1(self,x):
        if self.flat_attention:
            x=self.att1(x)
            assert x.shape[-1]==self.dim
            # x=torch.transpose(x,1,2)
            # torch.nn.BatchNorm1d(self.dim)
            # x=torch.transpose(x,1,2)
            nn.Softmax(dim=-1)
        else:
            x=self.att1(x)
        return x

    def cal_att2(self,x):
        if self.flat_attention:
            x=self.att2(x)
            # x=torch.transpose(x,1,2)
            # torch.nn.BatchNorm1d(self.dim)
            # x=torch.transpose(x,1,2)
            nn.Softmax(dim=-1)
        else:
            x=self.att2(x)
        return x

    
    def sim_func(self,a,b):
        if self.sim=='cosine':
            a=a/a.norm(dim=-1,keepdim=True)
            b=b/b.norm(dim=-1,keepdim=True)
            a=self.temperature*a
            b=self.temperature*b
            print('using cosine')

            return torch.mm(a,b.T)
        elif self.sim=='dot':
            print('using dot')

            return torch.mm(a,b.T)

    def compute_nce(self, encode_samples, pred):
        '''
        -----------------------------------------------------------------------------------
                        --------------Calculate NCE loss--------------
        -----------------------------------------------------------------------------------
        ...argument:
        ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
        ......pred :            Wk[i]( C_t )
        '''
        nce = 0  # average over time_step and batch
        self.batch_size = self.bs
        if self.pred_mode=='step':
            for i in np.arange(0, self.time_step):
                try:
                    print('self.sim is ',self.sim)
                    total = self.sim_func(encode_samples[i], pred[i])  # e.g. size 8*8
                except IndexError:
                    print('i is : %s,latent shape: %s, pred shape: %s ' % (i, encode_samples.shape, pred.shape))
                    raise AssertionError
                # print(total)
                
                if self.sym:
                    nce += 1/2*(torch.sum(torch.diag((nn.LogSoftmax(dim=0)(total)))) + torch.sum(torch.diag((nn.LogSoftmax(dim=1)(total)))))# nce is a tensor

                else:
                    nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
            nce /= -1. * self.batch_size * self.time_step
            accuracy = 1. * correct.item() / self.batch_size

        elif self.pred_mode=='future':
            total=self.sim_func(encode_samples[0],pred[0])
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            # correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
            #                                  torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            # correct_2=torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=1),
            #                                  torch.arange(0, self.batch_size).cuda())) 
            # print(correct,correct_2)                             
            # print(total)
            if self.sym:
                nce += 1/2*(torch.sum(torch.diag((nn.LogSoftmax(dim=0)(total)))) + torch.sum(torch.diag((nn.LogSoftmax(dim=1)(total)))))# nce is a tensor
            else:
                nce += torch.sum(torch.diag(self.lsoftmax(total)))   # nce is a tensor
            nce /= -1. * self.batch_size
            accuracy =1. * correct.item() / self.batch_size
        return nce, accuracy

    def encodeRegress(self, x, warm=False, conti=False):
        bs = x.shape[0]
        x = self.dropout(x)
        # print(x.shape)
        latents, state1 = self.lstm1(x)
        del x
        latents_to_pred = torch.mul(latents, self.cal_att1(latents))
        regs, state2 = self.lstm2(latents_to_pred)
        del latents_to_pred
        ht, ct = state2
        return latents, regs, ht, ct

    def get_reg_out(self, x, stack=False, warm=False, conti=False, ifbn=False):
        bs = x.shape[0]
        x = self.dropout(x)
        latents, state1 = self.lstm1(x)

        # latents_to_pred = torch.mul(latents, self.att1(latents))
        regs, state2 = self.lstm2(latents)
        ht, ct = state2
        return regs[:, -1, :].squeeze(1)

    def encode(self, x):
        bs = x.shape[0]
        x = self.dropout(x)
        x, (h, c) = self.lstm1(x)
        return x, h, c

    def decode(self, x):
        bs = x.shape[0]
        x = self.dropout(x)
        (h0, c0) = self.init_hidden(bs, self.input_dim)
        x, _ = self.lstm4(x, (h0, c0))
        return x

    def check_input(self, x):
        if type(x) == dict:
            dic = x
            x = dic['data'].squeeze(0)
            self.max_len = min(dic['length'])
            self.bs = x.shape[0]
        elif len(x.shape) == 4:
            x = x.squeeze(1)
            self.bs = x.shape[0]
            self.max_len = x.shape[1]
        elif x.shape[1] == 76:
            x = x.transpose(1, 2)
            self.bs = x.shape[0]
            self.max_len = x.shape[1]
        else:
            self.max_len=x.shape[1]
            self.bs=x.shape[0]
        return x

    def pred_future(self, x):
        x=self.check_input(x)

        # print(self.t_range)
        # print(self.max_len)
        t_range=(self.max_len*self.t_range[0],self.max_len*self.t_range[1])
        # print(t_range)
        # t_range = (self.max_len *2// 3, 4 * self.max_len // 5)
        # print(x.shape)
        x_ori = x
        if self.max_len>192: t=192
        else:
            t = torch.randint(low=int(t_range[0]), high=int(t_range[1]), size=(1,)).long()  # choose a point to split the time series
        # print('t is %s'%t)
        # self.bs = x.shape[0]
        latent_past, _, hidden_reg_out_past, _ = self.encodeRegress(x[:, :t + 1, :])
        latent_future, _, hidden_reg_out_future, _ = self.encodeRegress(x[:, t + 1:self.max_len, :])
        del x
        hidden_reg_out_pred = self.fcs(self.W_pred(hidden_reg_out_past))

        latent_all = torch.cat((latent_past, latent_future), 1)
        del latent_future,latent_past
        latent_all_attention = torch.mul(latent_all, self.cal_att2(latent_all))
        del latent_all
        x_hat = self.decode(latent_all_attention)
        nce, acc = self.compute_nce(self.fcs(hidden_reg_out_future), hidden_reg_out_pred)
        return x_hat, nce, acc

    def pred_timestep(self, x):
        x=self.check_input(x)
        t = torch.randint(low=20, high=self.max_len - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        # self.bs = x.shape[0]
        latent_past, _, hidden_reg_out, cell_reg_out = self.encodeRegress(x[:, :t + 1, :])
        latent_preds = self.fcs(
            self.predict(hidden_reg_out.transpose(0, 1), hidden_reg_out, cell_reg_out, self.time_step, self.mode))

        latent_future = []
        for i in range(1, self.time_step + 1):
            _, h, c = self.encode(x[:, t + i, :])
            latent_future.append(self.fcs(c[-1]))

        latent_future = torch.stack(latent_future, 0)

        latent_all = torch.cat((latent_past, latent_future.transpose(0, 1)), 1)

        latent_all_attention = torch.mul(latent_all, self.cal_att2(latent_all))
    
        x_hat = self.decode(latent_all_attention)
        nce, acc = self.compute_nce(latent_future, latent_preds)

        return x_hat, nce, acc

    def forward(self, x):
        if self.pred_mode == 'future':
            x_hat, nce, acc = self.pred_future(x)
        else:
            x_hat, nce, acc = self.pred_timestep(x)
        return x_hat, nce, acc

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.Wq = nn.Linear(in_dim , in_dim)
        self.Wk = nn.Linear(in_dim , in_dim)
        self.Wv = nn.Linear(in_dim , in_dim)
        self.gamma = in_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H) (batch_size X C X 76 X 192)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # x: (48, 144, 256)
        m_batchsize, width, height = x.size()
        proj_query = self.Wq(x)
        proj_key = self.Wk(x)
        energy = torch.matmul(proj_key.transpose(1,2),proj_query) / (self.gamma**0.5)
        attention = self.softmax(energy)
        proj_value = self.Wv(x)

        out = torch.matmul(proj_value, attention)


        return out


class CPAELSTM44_selfAT(CPLSTM4):
    """
    add decoder constraint in loss function
    pred_mode: 'step' for timestep prediction
                'future' for using past to predict future
    """

    def __init__(self, dim, bn, dropout, task,t_range=None,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5, mode=1, switch=True, pred_mode='step'):

        super(CPAELSTM44_selfAT, self).__init__(dim, bn, dropout, task,
                                            depth, num_classes,
                                            input_dim, time_step, mode, switch)

        self.lstm4 = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.input_dim,
            bidirectional=False,
            batch_first=True)
        self.att1 = SelfAttention(self.dim)
        self.att2 = SelfAttention(self.dim)
        self.t_range=t_range
        self.pred_mode = pred_mode


    def compute_nce(self, encode_samples, pred):
        '''
        -----------------------------------------------------------------------------------
                        --------------Calculate NCE loss--------------
        -----------------------------------------------------------------------------------
        ...argument:
        ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
        ......pred :            Wk[i]( C_t )
        '''
        nce = 0  # average over time_step and batch
        self.batch_size = self.bs
        if self.pred_mode=='step':
            for i in np.arange(0, self.time_step):
                try:
                    total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
                except IndexError:
                    print('i is : %s,latent shape: %s, pred shape: %s ' % (i, encode_samples.shape, pred.shape))
                    raise AssertionError
                # print(total)
                correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                             torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
                nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
            nce /= -1. * self.batch_size * self.time_step
            accuracy = 1. * correct.item() / self.batch_size
        elif self.pred_mode=='future':
            total=torch.mm(encode_samples[0],torch.transpose(pred[0], 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
            nce = torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
            nce /= -1. * self.batch_size
            accuracy =1. * correct.item() / self.batch_size
        return nce, accuracy

    def encodeRegress(self, x, warm=False, conti=False):
        bs = x.shape[0]
        x = self.dropout(x)
        latents, state1 = self.lstm1(x)
        del x
        # latents (48,144,256)
        latents_to_pred = self.att1(latents)
        regs, state2 = self.lstm2(latents_to_pred)
        del latents_to_pred
        ht, ct = state2
        return latents, regs, ht, ct

    def get_reg_out(self, x, stack=False, warm=False, conti=False, ifbn=False):
        bs = x.shape[0]
        x = self.dropout(x)
        latents, state1 = self.lstm1(x)

        # latents_to_pred = torch.mul(latents, self.att1(latents))
        regs, state2 = self.lstm2(latents)
        ht, ct = state2
        return regs[:, -1, :].squeeze(1)

    def encode(self, x):
        bs = x.shape[0]
        x = self.dropout(x)
        x, (h, c) = self.lstm1(x)
        return x, h, c

    def decode(self, x):
        bs = x.shape[0]
        x = self.dropout(x)
        (h0, c0) = self.init_hidden(bs, self.input_dim)
        x, _ = self.lstm4(x, (h0, c0))
        return x

    def check_input(self, x):
        if type(x) == dict:
            dic = x
            x = dic['data'].squeeze(0)
            self.max_len = min(dic['length'])
            self.bs = x.shape[0]
        elif len(x.shape) == 4:
            x = x.squeeze(1)
            self.bs = x.shape[0]
            self.max_len = x.shape[1]
        elif x.shape[1] == 76:
            x = x.transpose(1, 2)
            self.bs = x.shape[0]
            self.max_len = x.shape[1]
        else:
            self.max_len=x.shape[1]
            self.bs=x.shape[0]
        return x

    def pred_future(self, x):
        x=self.check_input(x)
        # print(self.t_range)
        # print(self.max_len)
        t_range=(self.max_len*self.t_range[0],self.max_len*self.t_range[1])
        # print(t_range)
        # t_range = (self.max_len *2// 3, 4 * self.max_len // 5)
        # print(x.shape)
        x_ori = x
        if self.max_len>192: t=192
        else:
            t = torch.randint(low=int(t_range[0]), high=int(t_range[1]), size=(1,)).long()  # choose a point to split the time series
        # print('t is %s'%t)
        # self.bs = x.shape[0]
        latent_past, _, hidden_reg_out_past, _ = self.encodeRegress(x[:, :t + 1, :])
        latent_future, _, hidden_reg_out_future, _ = self.encodeRegress(x[:, t + 1:self.max_len, :])
        del x
        hidden_reg_out_pred = self.fcs(self.W_pred(hidden_reg_out_past))

        latent_all = torch.cat((latent_past, latent_future), 1)
        del latent_future,latent_past
        latent_all_attention = self.att2(latent_all)
        del latent_all
        x_hat = self.decode(latent_all_attention)
        nce, acc = self.compute_nce(self.fcs(hidden_reg_out_future), hidden_reg_out_pred)
        return x_hat, nce, acc

    def pred_timestep(self, x):
        # x (48,192,76)
        x=self.check_input(x)
        # x (48,192,76)
        t = torch.randint(low=20, high=self.max_len - self.time_step - 1, size=(1,)).long()
        # print('reshape x to ',x.shape)
        # self.bs = x.shape[0]
        latent_past, _, hidden_reg_out, cell_reg_out = self.encodeRegress(x[:, :t + 1, :])
        latent_preds = self.fcs(
            self.predict(hidden_reg_out.transpose(0, 1), hidden_reg_out, cell_reg_out, self.time_step, self.mode))

        latent_future = []
        for i in range(1, self.time_step + 1):
            _, h, c = self.encode(x[:, t + i, :])
            latent_future.append(self.fcs(c[-1]))

        latent_future = torch.stack(latent_future, 0)

        latent_all = torch.cat((latent_past, latent_future.transpose(0, 1)), 1)
        latent_all_attention = self.att2(latent_all)
        x_hat = self.decode(latent_all_attention)
        nce, acc = self.compute_nce(latent_future, latent_preds)

        return x_hat, nce, acc

    def forward(self, x):
        if self.pred_mode == 'future':
            x_hat, nce, acc = self.pred_future(x)
        else:
            x_hat, nce, acc = self.pred_timestep(x)
        return x_hat, nce, acc

# class CPAELSTM45(CPLSTM4):
#     """
#     CPLSTM4+ CPAE4
#     """
#
#     def __init__(self, dim, bn, dropout, task,
#                  depth=2, num_classes=1,
#                  input_dim=76, time_step=5, mode=1):
#         super(CPAELSTM45, self).__init__(dim, bn, dropout, task,
#                                          depth, num_classes,
#                                          input_dim, time_step, mode)
#
#         self.fcs3 = nn.Sequential(
#             nn.Linear(self.input_dim, self.input_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.input_dim, self.input_dim)
#         )
#         self.lstm4 = nn.LSTM(
#             input_size=self.dim,
#             hidden_size=self.input_dim,
#             bidirectional=False,
#             batch_first=True)
#
#     def encode(self, x):
#         bs = x.shape[0]
#         x = self.dropout(x)
#         (h0, c0) = self.init_hidden(bs, self.dim)
#         x, _ = self.lstm1(x, (h0, c0))
#         return x
#
#     def decode(self, x):
#         bs = x.shape[0]
#         x = self.dropout(x)
#         (h0, c0) = self.init_hidden(bs, self.input_dim)
#         x, _ = self.lstm4(x, (h0, c0))
#         return x
#
#     def compute_nce(self, encode_samples, pred):
#         '''
#         -----------------------------------------------------------------------------------
#                         --------------Calculate NCE loss--------------
#         -----------------------------------------------------------------------------------
#         ...argument:
#         ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
#         ......pred :            Wk[i]( C_t )
#         '''
#         nce = 0  # average over time_step and batch
#         self.batch_size = self.bs
#         for i in np.arange(0, self.time_step):
#             try:
#                 total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
#             except IndexError:
#                 print('i is : %s,latent shape: %s, pred shape: %s ' % (i, encode_samples.shape, pred.shape))
#                 raise AssertionError
#             # print(total)
#             correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
#                                          torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
#             nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
#         nce /= -1. * self.batch_size * self.time_step
#         accuracy = 1. * correct.item() / self.batch_size
#
#         return nce, accuracy
#
#     def forward(self, x):
#         if len(x.shape) == 4: x = x.squeeze(1)
#         if x.shape[1] == 76: x = x.transpose(1, 2)
#         x_ori = x
#         t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
#         self.bs = x.shape[0]
#         xt, ht, ct = self.encodeRegress(x[:, :t + 1, :])
#
#         z_after_t = []
#
#         for i in range(1, self.time_step + 1):
#             _, h, c = self.encodeRegress(x[:, t + i, :])
#             z_after_t.append(c)
#         z_after_t = torch.cat(z_after_t, 0)
#
#         x_hat = self.decode(z_after_t)
#         nce, acc = self.compute_nce(self.fcs3(x_ori[:, t + 1:t + 1 + self.time_step, :]).transpose(0, 1),
#                                     self.fcs3(x_hat))
#         nce2, acc2 = self.compute_nce((x_ori[:, t + 1:t + 1 + self.time_step, :]).transpose(0, 1), x_hat)
#         print('acc after fc', acc)
#         print('acc before fc', acc2)
#         return acc, nce, None
#
#
# class CPAELSTM46(CPLSTM4):
#     """
#     CPLSTM4+ CPAE4
#     """
#
#     def __init__(self, dim, bn, dropout, task,
#                  depth=2, num_classes=1,
#                  input_dim=76, time_step=5, mode=1):
#         super(CPAELSTM46, self).__init__(dim, bn, dropout, task,
#                                          depth, num_classes,
#                                          input_dim, time_step, mode)
#
#         self.lstm4 = nn.LSTM(
#             input_size=self.dim,
#             hidden_size=self.input_dim,
#             bidirectional=False,
#             batch_first=True)
#
#     def encode(self, x):
#         bs = x.shape[0]
#         x = self.dropout(x)
#         (h0, c0) = self.init_hidden(bs, self.dim)
#         x, _ = self.lstm1(x, (h0, c0))
#         return x
#
#     def decode(self, x):
#         bs = x.shape[0]
#         x = self.dropout(x)
#         (h0, c0) = self.init_hidden(bs, self.input_dim)
#         x, _ = self.lstm4(x, (h0, c0))
#         return x
#
#     def compute_nce(self, encode_samples, pred):
#         '''
#         -----------------------------------------------------------------------------------
#                         --------------Calculate NCE loss--------------
#         -----------------------------------------------------------------------------------
#         ...argument:
#         ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
#         ......pred :            Wk[i]( C_t )
#         '''
#         nce = 0  # average over time_step and batch
#         self.batch_size = self.bs
#         for i in np.arange(0, self.time_step):
#             try:
#                 total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
#             except IndexError:
#                 print('i is : %s,latent shape: %s, pred shape: %s ' % (i, encode_samples.shape, pred.shape))
#                 raise AssertionError
#             # print(total)
#             correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
#                                          torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
#             nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
#         nce /= -1. * self.batch_size * self.time_step
#         accuracy = 1. * correct.item() / self.batch_size
#
#         return nce, accuracy
#
#     def forward(self, x):
#         if len(x.shape) == 4: x = x.squeeze(1)
#         if x.shape[1] == 76: x = x.transpose(1, 2)
#         x_ori = x
#         t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
#         self.bs = x.shape[0]
#         xt, ht, ct = self.encodeRegress(x[:, :t + 1, :])
#
#         z_after_t = []
#
#         for i in range(1, self.time_step + 1):
#             _, h, c = self.encodeRegress(x[:, t + i, :])
#             z_after_t.append(c)
#         z_after_t = torch.cat(z_after_t, 0)
#
#         x_hat = self.decode(z_after_t)
#         nce, acc = self.compute_nce((x_ori[:, t + 1:t + 1 + self.time_step, :]).transpose(0, 1), x_hat)
#         return acc, nce, None
#
#
# class CPAELSTM4_AT(CPLSTM4):
#     def __init__(self, dim, bn, dropout, task,
#                  depth=2, num_classes=1,
#                  input_dim=76, time_step=5, mode=1, switch=True):
#         super(CPAELSTM4_AT, self).__init__(dim, bn, dropout, task,
#                                            depth, num_classes,
#                                            input_dim, time_step, mode)
#
#         self.lstm1 = nn.LSTM(
#             input_size=self.input_dim,
#             hidden_size=self.dim,
#             num_layers=3,
#             bidirectional=False,
#             batch_first=True
#         )
#         self.lstm4 = nn.LSTM(
#             input_size=self.dim,
#             hidden_size=self.input_dim,
#             bidirectional=False,
#             batch_first=True)
#         self.switch = switch
#         if self.switch == False:
#             self.softmax = nn.Softmax(dim=1)
#             self.lsoftmax = nn.LogSoftmax(dim=1)
#         self.att1 = nn.Linear(self.dim, self.dim)  # attend to decoder
#         self.att2 = nn.Linear(self.dim, self.dim)  # attend to predictor
#
#     def encodeRegress(self, x, warm=False, conti=False):
#         bs = x.shape[0]
#         x = self.dropout(x)
#         latents, state1 = self.lstm1(x)
#         latents_to_pred = torch.mul(latents, self.att1(latents))
#         regs, state2 = self.lstm2(latents_to_pred)
#         ht, ct = state2
#         return latents, regs, ht, ct
#
#     def get_reg_out(self, x, stack=False, warm=False, conti=False, ifbn=False):
#         # TODO:
#         bs = x.shape[0]
#         x = self.dropout(x)
#         latents, state1 = self.lstm1(x)
#
#         # latents_to_pred = torch.mul(latents, self.att1(latents))
#         regs, state2 = self.lstm2(latents)
#         ht, ct = state2
#         return regs[:, -1, :].squeeze(1)
#
#     def compute_nce(self, encode_samples, pred):
#         '''
#         -----------------------------------------------------------------------------------
#                         --------------Calculate NCE loss--------------
#         -----------------------------------------------------------------------------------
#         ...argument:
#         ......encode_samples : ( time_step, batch_size, conv_sizes[-1] )
#         ......pred :            Wk[i]( C_t )
#         '''
#         nce = 0  # average over time_step and batch
#         self.batch_size = self.bs
#         for i in np.arange(0, self.time_step):
#             try:
#                 total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
#             except IndexError:
#                 print('i is : %s,latent shape: %s, pred shape: %s ' % (i, encode_samples.shape, pred.shape))
#                 raise AssertionError
#             # print(total)
#             correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
#                                          torch.arange(0, self.batch_size).cuda()))  # correct is a tensor
#             nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
#         nce /= -1. * self.batch_size * self.time_step
#         accuracy = 1. * correct.item() / self.batch_size
#
#         return nce, accuracy
#
#     def encode(self, x):
#         bs = x.shape[0]
#         x = self.dropout(x)
#         x, (h, c) = self.lstm1(x)
#         return x, h, c
#
#     def decode(self, x):
#         bs = x.shape[0]
#         x = self.dropout(x)
#         (h0, c0) = self.init_hidden(bs, self.input_dim)
#         x, _ = self.lstm4(x, (h0, c0))
#         return x
#
#     def forward(self, x):
#         # check shape
#         if len(x.shape) == 4: x = x.squeeze(1)
#         if x.shape[1] == 76: x = x.transpose(1, 2)
#         self.bs = x.shape[0]
#
#         # randomly choose a time point
#         t = torch.randint(low=20, high=x.shape[1] - self.time_step - 1, size=(1,)).long()
#
#         # encode the past and put into regressor
#         latent_past, _, hidden_reg_out, cell_reg_out = self.encodeRegress(x[:, :t + 1, :])
#         latent_preds = self.fcs(
#             self.predict(hidden_reg_out.transpose(0, 1), hidden_reg_out, cell_reg_out, self.time_step, self.mode))
#
#         latent_future = []
#         for i in range(1, self.time_step + 1):
#             _, h, c = self.encode(x[:, t + i, :])
#             latent_future.append(self.fcs(c[-1]))
#
#         latent_future = torch.stack(latent_future, 0)
#
#         latent_all = torch.cat((latent_past, latent_future.transpose(0, 1)), 1)
#         latent_all_attention = torch.mul(latent_all, self.att2(latent_all))
#         x_hat = self.decode(latent_all_attention)
#         nce, acc = self.compute_nce(latent_future, latent_preds)
#
#         return x_hat, nce, acc


class CDCK3_S(nn.Module):
    def __init__(
            self,
            embedded_features,
            gru_out,
            n_points=192,
            n_features=76,
            conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8],
            kernel_sizes=[(2, i) for i in [76, 32, 64, 64, 128, 256, 512, 1024, 512, 128, 64]],
            time_step=30):
        self.embedded_features = embedded_features
        self.gru_out = gru_out
        self.conv_sizes = conv_sizes
        self.time_step = time_step
        # kernel_sizes=get_kernel_sizes() #TODO
        super(CDCK3_S, self).__init__()
        self.n_features = n_features
        # . If is int, uses the same padding in all boundaries.
        #  If a 4-tuple, uses (left ,right ,top ,bottom )
        self.channels = [n_features] + conv_sizes

        # the core part of model list
        self.sequential = lambda inChannel, outChannel: nn.Sequential(
            nn.ReflectionPad1d((0, 1)),
            nn.Conv1d(inChannel, outChannel, kernel_size=2, padding=0),
            nn.BatchNorm1d(outChannel),
            nn.ReLU(inplace=True)
        )

        # ** minded the length should be 1 element shorter than # of channels
        self.encoder = nn.ModuleList(
            [self.sequential(self.channels[i], self.channels[i + 1]) for i in range(len(conv_sizes))]
        ).to(device)

        # self.decode_channels = self.channels[::-1]
        # self.decoder = nn.ModuleList(
        #     [self.sequential(self.decode_channels[i], self.decode_channels[i + 1]) for i in range(len(conv_sizes))]
        # ).to(device)
        self.linear = nn.Linear(self.conv_sizes[-1], self.embedded_features).to(device)

        self.Wk = nn.ModuleList([nn.Linear(self.gru_out, self.embedded_features) for i in range(self.time_step)]).to(
            device)

        # dim = 1 !!!
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.gru = nn.GRU(
            self.embedded_features,
            gru_out,
            num_layers=1,
            bidirectional=False,
            batch_first=True).to(device)
        self.beforeNCE = None

        # input shape: (N,C=1,n_points=192,n_features=76)
        # output shape: (N, C=sizes[-1], )
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(self._weights_init)

    def add_fcs(self, hidden=None):
        """
        This function will add FC layers to the embedded features and then compare the features after FC transformations.
        See NOTION for illustration.
        :param hidden: a list of hidden sizes per layer. For example:[100,100]. If no value is passed, it will be set
                    as [n_embedded_features,n_embedded_features]
        :return: None
        """
        n = self.embedded_features
        if hidden is None:
            self.fcs = nn.Sequential(
                nn.Linear(n, n),
                nn.ReLU(inplace=True),
                nn.Linear(n, n)
            )
        else:
            if type(hidden) != list:
                hidden = list(hidden)
            layers = []
            for i, j in zip([n] + hidden, hidden + [n]):
                layers.append(nn.Linear(i, j))
                layers.append(nn.ReLU(inplace=True))
            layers.pop()  # We do not want Relu at the last layer

            self.fcs = nn.Sequential(*layers).to(device)
        self.beforeNCE = True

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size, use_gpu=True):
        return torch.zeros(1, batch_size, self.gru_out).to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        # input shape: (N,C=1,n_points=192,n_features=76)

        if len(x.shape) == 4: x = x.squeeze(1)
        if x.shape[1] == 192: x = x.transpose(1, 2)
        for i in range(len(self.encoder)):  # input shape:  (N,n_features=76,n_points=192)
            x = self.encoder[i](x)

        # output shape: (N, C=conv_sizes[-1], n_frames,1)

        # output shape: (N, C=conv_sizes[-1], n_frames,1)

        self.n_frames = x.shape[2]
        t_samples = torch.randint(self.n_frames - self.time_step - 1, size=(1,)).long()
        encode_samples = torch.empty((self.time_step, batch_size, self.embedded_features)).float().to(
            device)  # e.g. size
        c_t = torch.zeros(size=(batch_size, self.gru_out)).float().to(device)
        hidden = self.init_hidden(batch_size, use_gpu=True)
        init_hidden = hidden
        # reshape for gru
        x = x.view(batch_size, self.n_frames, self.conv_sizes[-1])
        # output shape: (N,  n_frames, conv_sizes[-1])
        x = self.linear(x)
        # output shape: (N,  n_frames, embedded_features)
        for i in np.arange(1, self.time_step + 1):
            hidden = init_hidden
            encode_samples[i - 1, :, :] = x[:, int(t_samples) + i, :]
        forward_seq = x[:, :int(t_samples) + 1, :]
        # ----->SHAPE: (N,t_samples+1,embedded_features)
        output, hidden = self.gru(forward_seq, hidden)
        c_t = output[:, -1, :].view(batch_size, self.gru_out)

        pred = torch.empty((self.time_step, batch_size, self.embedded_features)).float().to(device)

        for i in np.arange(0, self.time_step):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        if self.beforeNCE:  # ADD FC layers
            pred = self.fcs(pred)
            encode_samples = self.fcs(encode_samples)

        #       -----------------------------------------------------------------------------------
        #       --------------Calculate NCE loss------------------------------------------------
        #       -----------------------------------------------------------------------------------
        nce = 0  # average over time_step and batch
        for i in np.arange(0, self.time_step):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            # print(total)
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, batch_size).to(device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch_size * self.time_step
        accuracy = 1. * correct.item() / batch_size

        return accuracy, nce, hidden

    def sub_forward(self, x):

        # input shape: (N,C=1,n_points=192,n_features=76)
        f = iter(self.convs)
        g = iter(self.bns)
        for i in range(len(self.conv_sizes)):
            x = next(f)(x)
            x = next(g)(x)
            x = nn.ReLU(inplace=True)(x)
            x = x.transpose(1, 3)
        return x

    def get_reg_out(self, x, every=False):
        batch_size = x.shape[0]
        # input shape: (N,C=1,n_points=192,n_features=76)

        if len(x.shape) == 4: x = x.squeeze(1)
        if x.shape[1] == 192: x = x.transpose(1, 2)
        for i in range(len(self.encoder)):  # input shape:  (N,n_features=76,n_points=192)
            x = self.encoder[i](x)
        # zt
        # output shape: (N, C=conv_sizes[-1], n_frames,1)
        self.n_frames = x.shape[2]
        t_samples = torch.randint(self.n_frames - self.time_step - 1, size=(1,)).long()
        encode_samples = torch.empty((self.time_step, batch_size, self.embedded_features)).float().to(
            device)  # e.g. size
        c_t = torch.zeros(size=(batch_size, self.gru_out)).float().to(device)
        hidden = self.init_hidden(batch_size)
        init_hidden = hidden

        # reshape for gru
        x = x.view(batch_size, self.n_frames, self.conv_sizes[-1])
        # output shape: (N,  n_frames, conv_sizes[-1])
        x = self.linear(x)
        # output shape: (N,  n_frames, embedded_features)

        hidden = init_hidden
        output, hidden = self.gru(x, hidden)
        c_t = output[:, -1, :].view(batch_size, self.gru_out)

        return c_t


class CDCK2(nn.Module):
    def __init__(self,
                 time_step,
                 batch_size,
                 frame_size,
                 fix_frame=True,
                 n_frames=None,
                 conv_sizes=[64, 128, 512, 128, 64, 32, 16],
                 n_flat_features_per_frame=None,
                 embedded_features=22,
                 gru_out=32
                 ):
        """data should be formatted as 
        Input: (batch size, n_frames, frame_size, features)
        *****If the frame_size and n_frames are identical for every batch, 
        *****Please set fix_frame=True, and please provide n_frames
        :type conv_sizes: list
        """

        super(CDCK2, self).__init__()
        self.beforeNCE = False

        self.frame_size = frame_size
        self.batch_size = batch_size
        self.time_step = time_step
        self.fix_frame = fix_frame
        self.n_frames = n_frames
        self.n_flat_features_per_frame = n_flat_features_per_frame
        self.embedded_features = embedded_features
        self.gru_out = gru_out
        if not self.fix_frame:
            self.encoder = nn.Sequential(
                nn.MaxPool2d(4, stride=1),
                nn.Conv2d(1, 4, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=1),
                nn.Conv2d(4, 8, kernel_size=2, stride=4, padding=2, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, self.embedded_features, kernel_size=2, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.embedded_features),
                nn.ReLU(inplace=True),
                nn.Flatten()
            )
        if self.fix_frame:
            self.convs = nn.ModuleList([nn.Conv2d(self.n_frames, conv_sizes[0], kernel_size=2, stride=1, padding=2,
                                                  bias=False, groups=self.n_frames)]
                                       + [
                                           nn.Conv2d(i, j, kernel_size=2, stride=1, padding=2, bias=False,
                                                     groups=self.n_frames)
                                           for i, j in zip(conv_sizes[:-1], conv_sizes[1:])
                                       ]
                                       )
            self.bns = nn.ModuleList(
                [nn.BatchNorm2d(i) for i in conv_sizes]
            )
            self.maxpooling = nn.MaxPool2d(2, stride=1)
            self.ReLU = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        if n_flat_features_per_frame:
            self.linear = nn.Linear(self.n_flat_features_per_frame, self.embedded_features)
            self.gru = nn.GRU(self.embedded_features, self.gru_out, num_layers=1, bidirectional=False,
                              batch_first=True).to(device)
            self.Wk = nn.ModuleList(
                [nn.Linear(self.gru_out, self.embedded_features) for i in range(self.time_step)]).to(
                device)

            # initialize gru
            for layer_p in self.gru._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(self._weights_init)

    def add_fcs(self, hidden=None):
        """
        This function will add FC layers to the embedded features and then compare the features after FC transformations.
        See NOTION for illustration.
        :param hidden: a list of hidden sizes per layer. For example:[100,100]. If no value is passed, it will be set
                    as [n_embedded_features,n_embedded_features]
        :return: None
        """
        n = self.embedded_features
        if hidden is None:
            self.fcs = nn.Sequential(
                nn.Linear(n, n),
                nn.ReLU(inplace=True),
                nn.Linear(n, n)
            )
        else:
            if type(hidden) != list:
                hidden = list(hidden)
            layers = []
            for i, j in zip([n] + hidden, hidden + [n]):
                layers.append(nn.Linear(i, j))
                layers.append(nn.ReLU(inplace=True))
            layers.pop()  # We do not want Relu at the last layer

            self.fcs = nn.Sequential(*layers)
        self.beforeNCE = True

    def update_flat_features(self, n_flat_features_per_frame):
        self.n_flat_features_per_frame = n_flat_features_per_frame
        self.linear = nn.Linear(self.n_flat_features_per_frame, self.embedded_features).to(device)
        self.gru = nn.GRU(self.embedded_features, self.gru_out, num_layers=1, bidirectional=False, batch_first=True).to(
            device)
        self.Wk = nn.ModuleList([nn.Linear(self.gru_out, self.embedded_features) for i in range(self.time_step)]).to(
            device)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size, use_gpu=True):
        if self.fix_frame:
            if use_gpu:
                return torch.zeros(1, batch_size, self.gru_out).to(device)
            else:
                return torch.zeros(1, batch_size, self.gru_out)
        if not self.fix_frame:
            if use_gpu:
                return torch.zeros(1, 1, self.gru_out).to(device)
            else:
                return torch.zeros(1, 1, self.gru_out)

    def forward(self, x):

        # Convert into frames
        # shape of x:(N,1,n_points,features)
        x, frame_ends = makeFrameDimension(x, self.frame_size,
                                           self.n_frames)  # shape of x:(batch_size,n_frames,frame_size, n_features)
        # shape of x:(N,n_frames,points_per_frame,features)

        batch_size = x.shape[0]
        # !warning!!!!! The last batch in the dataset may have batch_size < self.batch_size.
        # !!!!!!!!!!!!!! So cannot use self.batch_size here

        self.n_frames = x.shape[1]
        #       -----------------------------------------------------------------------------------
        #       --------------Pick a random time point------------------------------------------------
        #       -----------------------------------------------------------------------------------
        if not self.fix_frame:
            t_samples = torch.empty((batch_size, 1))
            for i in range(batch_size):
                try:
                    t_samples[i] = torch.randint(int((frame_ends[i] - self.time_step - 1).item()),
                                                 size=(1,)).long()  # randomly pick time stamps
                except RuntimeError:  # some patients have very few frames so we have to choose the first frame to start
                    frame_ends[i] = self.time_step + 3
                    t_samples[i] = 1

        if self.fix_frame:
            t_samples = torch.randint(self.n_frames - self.time_step - 1, size=(1,)).long()

        #       -----------------------------------------------------------------------------------
        #       --------------DO THE EMBEDDING------------------------------------------------
        #       ------------------------------------------------------------------------------------
        if not self.fix_frame:
            z = torch.empty((batch_size, self.n_frames, self.embedded_features)).float().to(device)
            for i in range(self.n_frames):
                y = (x[:, i, :, :].unsqueeze(1)).clone().to(device)
                y = self.encoder(y)  # ------>SHAPE: (N,n_flat_features_per_frame)

                # calculate n_flat_features_per_frame if it is unkown
                if self.n_flat_features_per_frame == None:
                    self.n_flat_features_per_frame = y.shape[1]
                    logger.info('-----n_flat_features_per_frame=%d' % self.n_flat_features_per_frame)
                    return self.n_flat_features_per_frame

                y = self.linear(y)  # ----->SHAPE: (N,embedded_features)
                z[:, i, :] = y.squeeze(1)  # --->SHAPE: (N,     1,   embedded_features)
            del x, y

        if self.fix_frame:
            # x:(8,24,8,76) (N,n_frames,points_per_frame,features)
            f = iter(self.convs)
            g = iter(self.bns)
            for i in range(len(self.convs)):
                x = next(f)(x)
                try:
                    x = nn.MaxPool2d(2, stride=2)(x)
                except RuntimeError:
                    pass
                x = next(g)(x)
                x = self.ReLU(x)
            x = nn.Flatten(start_dim=2, end_dim=-1)(x)
            z = x
            del x
            # z: (8,144)     (N,flat_features)

            # calculate n_flat_features_per_frame if it is unkown
            if self.n_flat_features_per_frame == None:
                self.n_flat_features_per_frame = int(z.shape[2] * z.shape[1] / self.n_frames)
                logger.info('-----n_flat_features_per_frame=%d' % self.n_flat_features_per_frame)
                return self.n_flat_features_per_frame

            z = z.view(batch_size, self.n_frames, self.n_flat_features_per_frame)
            # ---->SHAPE: (N,n_frames,n_flat_features_per_frame)
            z = self.linear(z)  # ----->SHAPE: (N,n_frames,embedded_features)

        encode_samples = torch.empty((self.time_step, batch_size, self.embedded_features)).float().to(
            device)  # e.g. size
        # ----->SHAPE: (T,N,embedded_features)

        c_t = torch.zeros(size=(batch_size, self.gru_out)).float().to(device)
        # output of GRU,------>SHAPE:(N, n_gru_out)

        #       -----------------------------------------------------------------------------------
        #       --------------GET GRU OUTPUT------------------------------------------------
        #       -----------------------------------------------------------------------------------

        forward_seq = []
        hidden = self.init_hidden(len(z), use_gpu=True)

        init_hidden = hidden

        if not self.fix_frame:
            for j in range(batch_size):
                hidden = init_hidden
                t = t_samples[j]
                for i in np.arange(1, self.time_step + 1):
                    encode_samples[i - 1][j] = z[j, int(t_samples[j].item()) + i, :]
                forward_seq.append(z[j, :int(t_samples[j].item()) + 1, :])
                output, hidden = self.gru(forward_seq[j].unsqueeze(0), hidden)
                c_t[j] = output[:, -1, :].view(1, self.gru_out)

        if self.fix_frame:
            for i in np.arange(1, self.time_step + 1):
                hidden = init_hidden
                encode_samples[i - 1, :, :] = z[:, int(t_samples) + i, :]
            forward_seq = z[:, :int(t_samples) + 1, :]
            # ----->SHAPE: (N,t_samples+1,embedded_features)
            output, hidden = self.gru(forward_seq, hidden)
            c_t = output[:, -1, :].view(batch_size, self.gru_out)

        pred = torch.empty((self.time_step, batch_size, self.embedded_features)).float().to(device)
        for i in np.arange(0, self.time_step):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        if self.beforeNCE:  # ADD FC layers
            pred = self.fcs(pred)
            encode_samples = self.fcs(encode_samples)
        #       -----------------------------------------------------------------------------------
        #       --------------Calculate NCE loss------------------------------------------------
        #       -----------------------------------------------------------------------------------
        nce = 0  # average over time_step and batch
        for i in np.arange(0, self.time_step):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            # print(total)
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                         torch.arange(0, batch_size).to(device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch_size * self.time_step
        accuracy = 1. * correct.item() / batch_size

        return accuracy, nce, hidden

    def get_reg_out(self, x, every=False):
        """
        Get the output of the regression model (GRU).
        batch_size could be different from the batch_size used in training process
        This function is only applicable for the case in which the samples share the same length,
        which means that the self.fix_frame=True
        """

        x, _ = makeFrameDimension(x, self.frame_size, x.shape[1])
        self.n_frames = x.shape[1]
        batch_size = x.size()[0]

        if self.fix_frame:
            f = iter(self.convs)
            g = iter(self.bns)
            for i in range(len(self.convs)):
                x = next(f)(x)
                try:
                    x = nn.MaxPool2d(2, stride=2)(x)
                except RuntimeError:
                    pass
                x = next(g)(x)
                x = self.ReLU(x)
            x = nn.Flatten(start_dim=2, end_dim=-1)(x)
            z = x
            # self.n_flat_features_per_frame=z.shape[1]/self.n_frames
            z = z.view(batch_size, self.n_frames, self.n_flat_features_per_frame)
            # ---->SHAPE: (N,n_frames,embedded_features)
            z = self.linear(z)  # ----->SHAPE: (N,n_frames,embedded_features)
            hidden = self.init_hidden(batch_size)
            output, hidden = self.gru(z, hidden)  # output size e.g. 8*128*256
            # ---->SHAPE: (N,n_frames,n_gru_out)





        else:
            z = torch.empty((batch_size, self.n_frames, self.embedded_features)).float().to(device)
            for i in range(self.n_frames):
                y = (x[:, i, :, :].unsqueeze(1)).clone().to(device)
                y = self.encoder(y)  # ------>SHAPE: (N,n_flat_features_per_frame)

                # calculate n_flat_features_per_frame if it is unkown
                if self.n_flat_features_per_frame == None:
                    self.n_flat_features_per_frame = y.shape[1]
                    logger.info('-----n_flat_features_per_frame=%d' % self.n_flat_features_per_frame)
                    return self.n_flat_features_per_frame

                y = self.linear(y)  # ----->SHAPE: (N,embedded_features)
                z[:, i, :] = y.squeeze(1)  # --->SHAPE: (N,     1,   embedded_features)
            del x, y

            c = torch.zeros(size=(batch_size, self.n_frames, self.gru_out)).float().to(device)
            for j in range(batch_size):
                hidden = self.init_hidden(batch_size)
                output, hidden = self.gru(z[j, :, :].unsqueeze(0), hidden)
                c[j, :, :] = output[:, :, :].view(1, self.n_frames, self.gru_out)
            output = c

        if every:
            return output  # return output from gru of every frame
        # ---->SHAPE: (N,n_frames,n_gru_out)
        else:
            return output[:, -1, :]  # only return the last output
        # ---->SHAPE: (N,n_gru_out)

    def get_latent(self, x, every=True):
        """
        Get the latent vectors of each frame
        """
        batch_size = x.size()[0]
        x, _ = makeFrameDimension(x, self.frame_size, x.shape[1])
        z = self.encoder(x)
        self.n_flat_features_per_frame = z.shape[1] / self.n_frames
        z = z.view(batch_size, self.n_frames, self.n_flat_features_per_frame)
        return z


class AE1(nn.Module):
    """
    trivial autoencoder
    """

    def __init__(
            self,
            conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8],
    ):
        super(AE1, self).__init__()
        self.conv_sizes = conv_sizes
        encodelist = []
        enChannels = [1] + conv_sizes
        count = 0
        for i in range(len(enChannels) - 1):
            encodelist.append(nn.Conv2d(enChannels[i], enChannels[i + 1], kernel_size=2))
            encodelist.append(nn.BatchNorm2d(enChannels[i + 1]))
            encodelist.append(nn.ReLU(inplace=True))
            # if count < 2:
            #     encodelist.append(nn.MaxPool2d(2,stride=1))
            count += 1
        deChannels = enChannels[::-1]
        decodelist = []
        for i in range(len(enChannels) - 1):
            # if count >= len(enChannels) - 3:
            #     decodelist.append(nn.ConvTranspose2d(deChannels[i], deChannels[i + 1], kernel_size=3))
            # else:
            decodelist.append(nn.ConvTranspose2d(deChannels[i], deChannels[i + 1], kernel_size=2))
            decodelist.append(nn.BatchNorm2d(deChannels[i + 1]))
            decodelist.append(nn.ReLU(inplace=True))
            count += 1

        self.encoder = nn.Sequential(*encodelist)
        self.decoder = nn.Sequential(*decodelist)

    def forward(self, x):
        y = x
        if len(x.shape) == 3: x.unsqueeze(1)
        x = self.encoder(x)
        # print(x.shape)
        torch.cuda.empty_cache()
        x = self.decoder(x)
        torch.cuda.empty_cache()

        # print(x.shape)
        if len(x.shape) == 4: x.squeeze(1)
        loss = nn.MSELoss(reduction='mean')(x, y)
        torch.cuda.empty_cache()

        return -1, loss, x  # make sure it is consistent with other models training function


class AE2_S(nn.Module):
    """
    Auto encoder, only move via time direction.  Same design in CPAE1
    """

    def __init__(
            self,
            embedded_features,
            conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8],
            n_points=192,
            n_features=76,
    ):
        self.conv_sizes = conv_sizes
        super(AE2_S, self).__init__()
        self.embedded_features = embedded_features
        # . If is int, uses the same padding in all boundaries.
        #  If a 4-tuple, uses (left ,right ,top ,bottom )
        self.channels = [n_features] + conv_sizes

        # the core part of model list
        self.sequential = lambda inChannel, outChannel: nn.Sequential(
            nn.ReflectionPad1d((0, 1)),
            nn.Conv1d(inChannel, outChannel, kernel_size=2, padding=0),
            nn.BatchNorm1d(outChannel),
            nn.ReLU(inplace=True)
        )

        # ** minded the length should be 1 element shorter than # of channels
        self.encoder = nn.ModuleList(
            [self.sequential(self.channels[i], self.channels[i + 1]) for i in range(len(conv_sizes))]
        )

        self.decode_channels = self.channels[::-1]
        self.decoder = nn.ModuleList(
            [self.sequential(self.decode_channels[i], self.decode_channels[i + 1]) for i in range(len(conv_sizes))]
        )
        self.linear = nn.Linear(self.conv_sizes[-1], self.embedded_features)
        self.delinear = nn.Linear(self.embedded_features, self.conv_sizes[-1])

    def forward(self, x):
        # input (batch,192,76)
        if len(x.shape) == 4: x = x.squeeze(1)
        y = x
        x = x.transpose(1, 2)  # (b,76,192)
        x = self.encode(x).transpose(1, 2)  # x: (batch, n_time, conv[-1])
        x = self.linear(x)  # (batch, time,embedded_features)
        x = nn.BatchNorm1d(self.embedded_features).to(device)(x.transpose(1, 2)).transpose(1, 2)
        x = nn.ReLU(inplace=True).to(device)(x)
        x = self.delinear(x)  # (batch, time, conv[-1])
        x = nn.BatchNorm1d(self.conv_sizes[-1]).to(device)(x.transpose(1, 2)).transpose(1, 2)
        x = nn.ReLU(inplace=True).to(device)(x)
        x = self.decode(x.transpose(1, 2))  # (batch,76,192)
        x = x.transpose(1, 2)
        loss = nn.MSELoss(reduction='mean')(x, y)
        return -1, loss, x

    def encode(self, x):
        for i in range(len(self.encoder)):  # input shape:  (N,n_features=76,n_points=192)
            x = self.encoder[i](x)
        return x  # output shape: (N,n_features=8,n_points=192)

    def decode(self, x):
        for i in range(len(self.decoder)):  # input shape:   (N,n_features=8,n_points=192)
            x = self.decoder[i](x)
        return x

    def get_encode(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        x = x.transpose(1, 2)
        x = self.encode(x).transpose(1, 2)
        x = nn.Flatten()(x)
        return x  # output shape: (N,192*12)


class CAE1(AE1):
    """
    Contrastive Auto-encoder based on AE1
    """

    def __init__(self):
        super(CAE1, self).__init__()
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        # get batch size
        bs = x.shape[0]
        y = x
        _, _, x = super().forward(x)
        loss, acc = self.compute_nce(x, y)
        del y
        return acc, loss, x

    def compute_nce(self, x_hat, x):
        bs = x.shape[0]
        assert x.shape == x_hat.shape
        nce = 0
        x = x.view(bs, -1)
        x_hat = x_hat.view(bs, -1)
        total = torch.mm(x_hat, x.T)
        correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                     torch.arange(0, bs).cuda()))
        nce = torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * bs
        acc = 1. * correct.item() / bs
        torch.cuda.empty_cache()

        del x, x_hat

        return nce, acc


class CAE11(nn.Module):
    def __init__(
            self,
            conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8],
    ):
        super(CAE11, self).__init__()
        self.conv_sizes = conv_sizes
        encodelist = []
        enChannels = [1] + conv_sizes
        count = 0
        for i in range(len(enChannels) - 1):
            encodelist.append(nn.Conv2d(enChannels[i], enChannels[i + 1], kernel_size=2))
            encodelist.append(nn.BatchNorm2d(enChannels[i + 1]))
            encodelist.append(nn.ReLU(inplace=True))
            # if count < 2:
            #     encodelist.append(nn.MaxPool2d(2,stride=1))
            count += 1
        deChannels = enChannels[::-1]
        decodelist = []
        for i in range(len(enChannels) - 1):
            # if count >= len(enChannels) - 3:
            #     decodelist.append(nn.ConvTranspose2d(deChannels[i], deChannels[i + 1], kernel_size=3))
            # else:
            decodelist.append(nn.ConvTranspose2d(deChannels[i], deChannels[i + 1], kernel_size=2))
            decodelist.append(nn.BatchNorm2d(deChannels[i + 1]))
            decodelist.append(nn.ReLU(inplace=True))
            count += 1

        self.encoder = nn.Sequential(*encodelist)
        self.decoder = nn.Sequential(*decodelist)

        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        y = x
        if len(x.shape) == 3: x.unsqueeze(1)
        x = self.encoder(x)
        # print(x.shape)
        torch.cuda.empty_cache()
        x = self.decoder(x)
        torch.cuda.empty_cache()
        # print(x.shape)
        if len(x.shape) == 4: x.squeeze(1)
        torch.cuda.empty_cache()
        loss, acc = self.compute_nce(x, y)
        del y
        return acc, loss, x

    def compute_nce(self, x_hat, x):
        bs = x.shape[0]
        assert x.shape == x_hat.shape
        nce = 0
        x = x.view(bs, -1)
        x_hat = x_hat.view(bs, -1)
        total = torch.mm(x_hat, x.T)
        correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                     torch.arange(0, bs).cuda()))
        nce = torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * bs
        acc = 1. * correct.item() / bs
        torch.cuda.empty_cache()

        del x, x_hat

        return nce, acc


class CAE2_S(AE2_S):
    """
    Contrastive auto-encoder based on AE2
    """

    def __init__(
            self,
            embedded_features,
            conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8],
            n_points=192,
            n_features=76,
    ):
        self.conv_sizes = conv_sizes
        self.embedded_features = embedded_features

        super(CAE2_S, self).__init__(self.embedded_features, self.conv_sizes)
        # # . If is int, uses the same padding in all boundaries.
        # #  If a 4-tuple, uses (left ,right ,top ,bottom )
        # self.channels = [n_features] + conv_sizes
        #
        # # the core part of model list
        # self.sequential = lambda inChannel, outChannel: nn.Sequential(
        #     nn.ReflectionPad1d((0, 1)),
        #     nn.Conv1d(inChannel, outChannel, kernel_size=2, padding=0),
        #     nn.BatchNorm1d(outChannel),
        #     nn.ReLU(inplace=True)
        # )
        #
        # # ** minded the length should be 1 element shorter than # of channels
        # self.encoder = nn.ModuleList(
        #     [self.sequential(self.channels[i], self.channels[i + 1]) for i in range(len(conv_sizes))]
        # )
        #
        # self.decode_channels = self.channels[::-1]
        # self.decoder = nn.ModuleList(
        #     [self.sequential(self.decode_channels[i], self.decode_channels[i + 1]) for i in range(len(conv_sizes))]
        # )
        # self.linear = nn.Linear(self.conv_sizes[-1], self.embedded_features)
        # self.delinear = nn.Linear(self.embedded_features, self.conv_sizes[-1])
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        y = x
        x = x.transpose(1, 2)  # (b,76,192)
        x = self.encode(x).transpose(1, 2)  # x: (batch, n_time, conv[-1])
        x = self.linear(x)  # (batch, time,embedded_features)
        x = nn.BatchNorm1d(self.embedded_features).to(device)(x.transpose(1, 2)).transpose(1, 2)
        x = nn.ReLU(inplace=True).to(device)(x)
        x = self.delinear(x)  # (batch, time, conv[-1])
        x = nn.BatchNorm1d(self.conv_sizes[-1]).to(device)(x.transpose(1, 2)).transpose(1, 2)
        x = nn.ReLU(inplace=True).to(device)(x)
        x = self.decode(x.transpose(1, 2))  # (batch,76,192)
        x = x.transpose(1, 2)
        loss, acc = self.compute_nce(x, y)  # TODO:
        return acc, loss, x

    def compute_nce(self, x_hat, x):
        bs = x.shape[0]
        assert x.shape == x_hat.shape
        nce = 0
        x = x.view(bs, -1)
        x_hat = x_hat.reshape(bs, -1)
        total = torch.mm(x_hat, x.T)
        correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                     torch.arange(0, bs).cuda()))
        nce = torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * bs
        acc = 1. * correct.item() / bs

        return nce, acc


class Basic_Cnn(nn.Module):
    def __init__(self, seed, conv_sizes=[32, 64, 64, 128, 256, 512, 1024, 512, 128, 64, 8], n_features=76, out=2):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        super(Basic_Cnn, self).__init__()
        torch.manual_seed(seed)

        # . If is int, uses the same padding in all boundaries.
        #  If a 4-tuple, uses (left ,right ,top ,bottom )
        self.out = out
        self.channels = [n_features] + conv_sizes

        # the core part of model list
        self.sequential = lambda inChannel, outChannel: nn.Sequential(
            nn.ReflectionPad1d((0, 1)),
            nn.Conv1d(inChannel, outChannel, kernel_size=2, padding=0),
            nn.BatchNorm1d(outChannel),
            nn.ReLU(inplace=True)
        )

        # ** minded the length should be 1 element shorter than # of channels
        self.encoder = nn.ModuleList(
            [self.sequential(self.channels[i], self.channels[i + 1]) for i in range(len(conv_sizes))]
        )

        self.fc = nn.Sequential(
            nn.Linear(self.channels[-1], 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.out),
            nn.LogSoftmax(dim=1)
        )

        # dim = 1 !!!
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)

        # input shape: (N,C=1,n_points=192,n_features=76)
        # output shape: (N, C=sizes[-1], )

        self.apply(self._weights_init)
        # def relevant_points(n):

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(len(self.encoder)):  # input shape:  (N,n_features=76,n_points=192)
            x = self.encoder[i](x)  # ouput shape:  (N,8,192)
        y = self.fc(x[:, :, -1])
        return y


def train(args, model, device, train_loader, optimizer, epoch, batch_size, lr=None):
    # turn on the training mode
    model.train()
    logger = logging.getLogger("cpc")

    if 'CPAE' not in args['model_type'] or 'CPAE4' in args['model_type'] or (
            'CPAE7' in args['model_type']) or 'CPAELSTM41' in args['model_type'] or 'CPAELSTM42' in args['model_type']:
        for batch_idx, sample in enumerate(train_loader):
            if sample == 1: continue
            sigs, labels = zip(*sample)
            sigs = torch.stack(sigs)
            labels = torch.stack(labels)
            data = sigs.float().unsqueeze(1).to(device)  # add channel dimension
            data.requires_grad = True
            optimizer.zero_grad()

            # If n_flat_features_per_frame is not provided, then the forward() of the above sentence will return
            # n_flat_features_per_frame and the below sentence will raise TypeError.
            # Then get the n_flat_features_per_frame and update this to the model
            # DO the forward again

            result = model(data)
            try:
                acc, loss, hidden = result
            except TypeError:
                n_flat_features_per_frame = result
                return result

            loss.backward()
            optimizer.step()

            if lr is None:
                lr = optimizer.update_learning_rate()  # See optimizer.py
                # print(lr)
            if batch_idx % args['log_interval'] == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), lr, acc, loss.item()))

            del sigs, labels, sample, data, hidden, acc, loss
            torch.cuda.empty_cache()

    elif 'CPAE' in args['model_type']:

        model.train()
        logger.info('\n --------------------------- epoch {} ------------------------- \n'.format(epoch))
        if args.get('lambda'): logger.info('weights are %s' % args['lambda'])
        local_loss = []
        for ii, batch in enumerate(train_loader):
            if batch == 1:
                continue
            X, y = zip(*batch)
            X = torch.stack(X).to(device)
            X.requires_grad = True
            # y = torch.tensor(y).long().to('cuda') # y is not used here in autoencoder

            optimizer.zero_grad()

            D, nce, accuracy = model(X)  # decoded
            l = args.get('Lambda')
            if l:
                loss = Chimera_loss(D, X, nce, l)
            else:
                loss = Chimera_loss(D, X, nce)

            loss.backward()
            optimizer.step()

            local_loss.append(loss.item())

            if ii % 100 == 0:  # verbose
                new_lr = optimizer.update_learning_rate()
                logger.info('\t {:.5f} {:.5f}'.format(loss.item(), new_lr))
            del X, y, batch, D, nce, accuracy, loss, ii
            torch.cuda.empty_cache()

        logger.info('\n ---------------------- mean loss : {:.5f}  ---------------------- \n'.format(
            np.mean(local_loss)))
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()


def validation(model, args, device, validation_loader):
    logger = logging.getLogger("cpc")

    logger.info("Starting Validation")
    if 'CPAE' not in args['model_type'] or 'CPAELSTM42' in args['model_type'] or ('CPAE4' in args['model_type']) or (
            'CPAE7' in args['model_type']) or 'CPAELSTM41' in args['model_type']:
        model.eval()
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for _, sample in enumerate(validation_loader):
                if sample == 1: continue
                sigs, _ = zip(*sample)
                sigs = torch.stack(sigs)
                data = sigs.float().unsqueeze(1).to(device)
                acc, loss, hidden = model(data)
                total_loss += len(data) * loss
                total_acc += len(data) * acc
                torch.cuda.empty_cache()
                del sigs, sample
            return total_acc, total_loss
    else:
        model.eval()
        loss_ls = []
        total_loss = 0
        total_acc = 0
        for ii, batch in enumerate(validation_loader):
            if batch == 1: continue
            X, y = zip(*batch)
            X = torch.stack(X).to('cuda')

            D, nce, accuracy = model(X)  # decoded
            if args.get('lambda'):
                total_loss += Chimera_loss(D, X, nce, args['lambda']).detach().cpu().numpy()
            else:
                total_loss += Chimera_loss(D, X, nce).detach().cpu().numpy()

            loss_ls.append(record_loss(D, X, nce))
            total_acc += len(X) * accuracy
            torch.cuda.empty_cache()
            del X, y, batch, D, nce, accuracy

        loss_ls = np.stack(loss_ls)
        logger.info('\n                     ------- validation -------            \n'.format(ii))
        logger.info('\t NCE \t MSE \t MASK MSE \t MAPPING MSE')
        logger.info('\t {:.4f}  \t {:.4f}  \t {:.4f} \t {:.4f}'.format(*np.mean(loss_ls, axis=0)))

        return total_acc, total_loss


def define_model(args_json, Model, train_loader):
    model_args = filter_args(args_json, Model)
    model = Model(**model_args)
    optimizer = eval(args_json['optimizer'])

    if args_json.get('n_flat_features_per_frame') is None and Model == CDCK2:
        args_json['n_flat_features_per_frame'] = train(args_json, model, device, train_loader, optimizer, 2,
                                                       args_json['batch_size'])
        del model
        model_args = filter_args(args_json, Model)
        model = Model(**model_args)
        model.update_flat_features(args_json['n_flat_features_per_frame'])
    if args_json.get('fcs') is not None:
        model.add_fcs(args_json['fcs'])  # add fc layers if required
    return model.to(device), optimizer


def save_intermediate(Model, args_json, device):
    setting_name = get_setting_name(args_json['model_best'])
    logging_dir = args_json['logging_dir']
    checkpoint_path = os.path.join(
        args_json['top_path'],
        'logs/cpc/',
        args_json['model_type'],
        args_json['model_best']
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print('Starting to generate intermediate data\n')
    train_loader, validation_loader, test_loader = split_Structure_Inhospital(
        args_json, percentage=1)  # BUG every data sample is the same!!!
    model, optimizer = define_model(args_json, Model, train_loader)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    context_train = []
    context_val = []
    context_test = []
    y_train = []
    y_test = []
    y_val = []
    model.eval()

    with torch.no_grad():
        for _, sample in enumerate(train_loader):
            if sample == 1: break
            x, y = zip(*sample)
            out = model.get_reg_out(
                (
                    torch.stack(x).float().unsqueeze(1).to(device)
                )
            ).cpu()
            context_train.append(out)
            torch.cuda.empty_cache()
            y_train.append((torch.stack(y)))
            del sample, x, y, out
        context_train = torch.cat(context_train).cpu().numpy()
        y_train = torch.cat(y_train).cpu().numpy()
        np.save(os.path.join(logging_dir, setting_name + '-x_train'), context_train)
        np.save(os.path.join(logging_dir, setting_name + '-y_train'), y_train)
        print('Getting training intermediate vectors done. saved in %s' % logging_dir)
        torch.cuda.empty_cache()
        del context_train, y_train

        for _, sample in enumerate(validation_loader):
            if sample == 1: break

            x, y = zip(*sample)
            context_val.append(model.get_reg_out(
                (
                    torch.stack(
                        x
                    ).float().unsqueeze(1).to(device)
                )
            )
            )
            y_val.append((torch.stack(y)))
            del sample, x, y
        context_val = torch.cat(context_val).cpu().numpy()
        y_val = torch.cat(y_val).cpu().numpy()
        np.save(os.path.join(logging_dir, setting_name + '-x_val'), context_val)
        np.save(os.path.join(logging_dir, setting_name + '-y_val'), y_val)
        print('Getting validation intermediate vectors done. saved in %s' % logging_dir)
        torch.cuda.empty_cache()
        del context_val, y_val

        for _, sample in enumerate(test_loader):
            if sample == 1: break

            x, y = zip(*sample)
            context_test.append(model.get_reg_out(
                (
                    torch.stack(
                        x
                    ).float().unsqueeze(1).to(device)
                )
            )
            )
            y_test.append((torch.stack(y)))
            del sample, x, y
        context_test = torch.cat(context_test).cpu().numpy()
        y_test = torch.cat(y_test).cpu().numpy()
        np.save(os.path.join(logging_dir, setting_name + '-x_test'), context_test)
        np.save(os.path.join(logging_dir, setting_name + '-y_test'), y_test)
        print('Getting test intermediate vectors done. saved in %s' % logging_dir)
        torch.cuda.empty_cache()
        del context_test, y_test


def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                                 run_name + '-model_best.pth')
    # torch.save can save any object
    # dict type object in our cases
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))


def my_collate(batch):
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
    if len(batch) > 1:
        data = []
        labels = []
        durations = []
        batch = sorted(batch, key=lambda x: x['duration'], reverse=True)
        for sample in batch:
            data.append(sample['patient'])
            labels.append(sample['death'])
            durations.append(sample['duration'])
        max_len, n_feats = data[0].shape
        data = [np.array(s, dtype=float) for s in data]
        data = [torch.from_numpy(s).float() for s in data]
        labels = [label for label in labels]
        durations = [duration for duration in durations]
        data = [torch.cat((s, torch.zeros(max_len - s.shape[0], n_feats)), 0) if s.shape[0] != max_len else s for s in
                data]
        data = torch.stack(data, 0)  # shape:[24,2844,462]
        labels = torch.stack(labels, 0)
        durations = torch.stack(durations, 0)  # max:2844
    return data, labels, durations


class MLP(nn.Module):
    def __init__(self, hidden_sizes, seed, in_features=8, out=2, dropout=True):
        torch.manual_seed(seed)
        super(MLP, self).__init__()
        hidden_sizes = [in_features] + hidden_sizes + [out]
        l = []
        torch.manual_seed(seed)

        fcs = [nn.Linear(i, j, bias=True) for i, j in zip(hidden_sizes[:-1], hidden_sizes[1:])]
        relu = nn.ReLU(inplace=True)
        drop = nn.Dropout(p=0.2)
        torch.manual_seed(seed)

        bns = [nn.BatchNorm1d(i) for i in hidden_sizes[1:]]
        # apply(_weights_init)
        for i in range(len(hidden_sizes) - 1):
            l.append(fcs[i])
            if i != len(hidden_sizes) - 2:
                l.append(relu)
                l.append(bns[i])
                if dropout: l.append(drop)
        self.mymodules = nn.Sequential(*l)

        for model in self.mymodules:
            self.initialize_weights(model)

    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)

    def forward(self, x):
        # print(x.shape)
        if len(x.shape) == 4:
            x = x.squeeze(1)  # fastai has a strange issue here.
        x = self.mymodules(x)
        # print (x)
        # print(x.shape)
        return x

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def valid(self, data_loader, iterations='all', metrics=None):
        if metrics == None: metrics = self.metrics
        loss = [None] * len(metrics)
        overall_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if iterations != 'all':
                    if i >= iterations: return overall_loss
                ct, y = zip(*batch)
                ct = torch.stack(ct).squeeze(1).to(device)

                y = torch.stack(y).cpu()
                pred = self.model(ct).cpu()  # forward
                for i, metric in enumerate(metrics):
                    loss[i] = metric(pred, y)  # loss
                overall_loss.append((loss))
                del loss, ct, y, pred

        return overall_loss


class LR(nn.Module):
    def __init__(self, seed, in_features=8, out=2):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        super(LR, self).__init__()
        torch.manual_seed(seed)

        self.linear = nn.Linear(in_features, out)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=1)


def load_intermediate(top_path, setting_name, model_type):
    middata_dir = os.path.join(top_path, 'logs', 'imp', model_type)
    x_train = np.load(os.path.join(middata_dir, setting_name + '-x_train.npy'))
    y_train = np.load(os.path.join(middata_dir, setting_name + '-y_train.npy'))
    x_val = np.load(os.path.join(middata_dir, setting_name + '-x_val.npy'))
    y_val = np.load(os.path.join(middata_dir, setting_name + '-y_val.npy'))
    x_test = np.load(os.path.join(middata_dir, setting_name + '-x_test.npy'))
    y_test = np.load(os.path.join(middata_dir, setting_name + '-y_test.npy'))
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test
    }


def tabular_frame(args_json):
    data_intermediate = load_intermediate(args_json['top_path'], args_json['setting_name'], args_json['model_type'])
    x_train, y_train, x_val, y_val, x_test, y_test = data_intermediate['x_train'], data_intermediate['y_train'], \
                                                     data_intermediate['x_val'], data_intermediate['y_val'], \
                                                     data_intermediate['x_test'], data_intermediate['y_test']
    train_df = pd.DataFrame(np.hstack((x_train, y_train)), columns=list(range(8)) + ['y'])
    val_df = pd.DataFrame(np.hstack((x_val, y_val)), columns=list(range(8)) + ['y'])
    test_df = pd.DataFrame(np.hstack((x_test, y_test)), columns=list(range(8)) + ['y'])
    return train_df, val_df, test_df


def dataset_intermediate(args_json):
    data_intermediate = load_intermediate(args_json['top_path'], args_json['setting_name'], args_json['model_type'])
    x_train, y_train, x_val, y_val, x_test, y_test = data_intermediate['x_train'], data_intermediate['y_train'], \
                                                     data_intermediate['x_val'], data_intermediate['y_val'], \
                                                     data_intermediate['x_test'], data_intermediate['y_test']
    train_set, val_set, test_set = TrivialDataset(x_train, y_train), \
                                   TrivialDataset(x_val, y_val), \
                                   TrivialDataset(x_test, y_test)

    return train_set, val_set, test_set


def data_loader_intermediate(args_json):
    data_intermediate = load_intermediate(args_json['top_path'], args_json['setting_name'], args_json['model_type'])
    x_train, y_train, x_val, y_val, x_test, y_test = data_intermediate['x_train'], data_intermediate['y_train'], \
                                                     data_intermediate['x_val'], data_intermediate['y_val'], \
                                                     data_intermediate['x_test'], data_intermediate['y_test']
    train_set, val_set, test_set = TrivialDataset(x_train, y_train), \
                                   TrivialDataset(x_val, y_val), \
                                   TrivialDataset(x_test, y_test)
    train_loader, val_loader, test_loader = DataLoader(train_set, shuffle=True, batch_size=args_json['batch_size'],
                                                       collate_fn=my_collate_fix,
                                                       num_workers=args_json['num_workers']), \
                                            DataLoader(val_set, batch_size=args_json['batch_size'], shuffle=True,
                                                       collate_fn=my_collate_fix,
                                                       num_workers=args_json['num_workers']), \
                                            DataLoader(test_set, shuffle=False, batch_size=args_json['batch_size'],
                                                       collate_fn=my_collate_fix,
                                                       num_workers=args_json['num_workers'])

    return train_loader, val_loader, test_loader


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def fastai_dl(train_set, val_set, test_set, device, batch_size=64, num_workers=24):
    # fastai dataloader
    return tabular.DataBunch.create(train_ds=train_set, valid_ds=val_set, test_ds=test_set,
                                    bs=batch_size, num_workers=num_workers, device=device,
                                    )


def train_mlp(model, train_loader, val_loader, epoch, lr, optimizer):
    lossfn = nn.CrossEntropyLoss()
    for epoch in range(epoch):

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        model.train()
        for i, batch in enumerate(train_loader):
            ct, y = zip(*batch)
            ct = torch.stack(ct).squeeze(1).to(device)
            y = torch.stack(y).to(device)
            # ---------- train  mlp ---------
            optimizer.zero_grad()

            pred = model(ct)  # forward
            loss = lossfn(pred, y)  # loss
            acc = sum(torch.eq(torch.argmax(pred, axis=1), y)).item() / len(y) * 100

            train_acc.append(acc)
            loss.backward()  # compute loss
            optimizer.step()  # update
            torch.cuda.empty_cache()
            train_loss.append(loss.item())

            del pred, loss, acc, ct, y

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                ct, y = zip(*batch)
                ct = torch.stack(ct).squeeze(1).to(device)

                y = torch.stack(y).to(device)
                # ---------- validation predicted by  mlp ---------
                pred = model(ct)  # forward
                loss = lossfn(pred, y)  # loss
                acc = sum(torch.eq(torch.argmax(pred, axis=1), y)).item() / len(y) * 100

                val_acc.append(acc)
                val_loss.append(loss.item())

                torch.cuda.empty_cache()

                del pred, loss, acc, ct, y

        # print out statistics
        verbose(epoch, train_loss, train_acc, val_loss, val_acc)


class Basic_LSTM(nn.Module):
    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5, mode=1, noct=False):

        self.out = 2 if task in ['ihm', 'dd'] else 10

        super(Basic_LSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.out),
            nn.LogSoftmax(dim=1)
        )

        for model in [self.lstm1, self.fc]:
            self.initialize_weights(model)

    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

    def forward(self, x):
        xt, state1 = self.lstm1(x)

        y = self.fc(xt[:, -1, :])
        return y


class AE_LSTM(nn.Module):
    """
    CPLSTM4------use lstm as Wk
    mode=1 use hidden states when predict. else use cell states
    """

    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5, mode=1, noct=False):
        self.dim = dim  # hidden dimension
        self.bn = bn
        self.drop = dropout
        self.task = task
        self.depth = depth
        self.time_step = time_step
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.mode = mode
        self.noct = noct
        super(AE_LSTM, self).__init__()

        # encoder
        self.lstm1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=dim,
            bidirectional=False,
            batch_first=True
        )

        # decoder   
        # minded that hidden_size is different
        self.lstm2 = nn.LSTM(
            input_size=dim,
            hidden_size=self.input_dim,
            bidirectional=False,
            batch_first=True
        )

        # not used

        if self.noct:
            self.stack_dim = self.dim * 192
        else:
            self.stack_dim = self.dim * 193
        self.dropout = nn.Dropout(self.drop)
        # self.Wk = nn.ModuleList([nn.Linear(self.dim, self.dim) for i in range(self.time_step)])
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)

        for model in [self.lstm1, self.lstm2]:
            self.initialize_weights(model)

    def init_hidden(self, bs, dim):
        cell_states = torch.zeros(1, bs, dim).to(device)
        hidden_states = torch.zeros(1, bs, dim).to(device)
        return (hidden_states, cell_states)

    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

    def get_reg_out(self, x, stack=False, warm=False, conti=False):
        # check input shape
        if len(x.shape) == 4: x = x.squeeze(1)
        if x.shape[1] == 76: x = x.transpose(1, 2)

        xt, (ht, ct) = self.lstm1(x)

        if stack and self.noct: return self.dropout(xt.reshape((x.shape[0], -1)))

        if stack: return self.dropout(torch.cat((xt.reshape((x.shape[0], -1)), ct.squeeze(0)), 1))
        return xt[:, -1, :].squeeze(1)

    def get_encode(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        if x.shape[1] == 76: x = x.transpose(1, 2)
        x,_,_ = self.lstm1(x)
        x = nn.Flatten()(x)
        return x

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)

        self.bs = x.shape[0]

        x_t, state1 = self.lstm1(x)  # encoder part : zeros init
        x_hat, state2 = self.lstm2(x_t)  # decoder part : zeros init

        loss = nn.MSELoss(reduction='mean')(x, x_hat)

        return -1, loss, x  # make sure it is consistent with other models training function


class CAE_LSTM(AE_LSTM):
    """
    constrastive auto-encoder with LSTM backbone
    """

    def __init__(self, dim, bn, dropout, task,
                 depth=2, num_classes=1,
                 input_dim=76, time_step=5, mode=1, noct=False):
        super(CAE_LSTM, self).__init__(dim, bn, dropout, task, depth, num_classes, input_dim, time_step, mode, noct)
        # get reg out is also the same  as Basic LSTM_AE

    def forward(self, x):
        if len(x.shape) == 4: x = x.squeeze(1)
        # print('shape of x is ' ,x.shape)
        if x.shape[1] == 76: x = x.transpose(1, 2)

        self.bs = x.shape[0]

        x_t, state1 = self.lstm1(x)  # encoder part : zeros init
        x_hat, state2 = self.lstm2(x_t)  # decoder part : zeros init

        loss, acc = self.compute_nce(x_hat, x)
        return acc, loss, x

    def compute_nce(self, x_hat, x):
        bs = x.shape[0]
        assert x.shape == x_hat.shape
        nce = 0
        x = x.view(bs, -1)
        x_hat = x_hat.reshape(bs, -1)
        total = torch.mm(x_hat, x.T)
        correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                     torch.arange(0, bs).cuda()))
        nce = torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * bs
        acc = 1. * correct.item() / bs

        return nce, acc
