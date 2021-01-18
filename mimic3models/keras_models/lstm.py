from __future__ import absolute_import
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout
from keras.layers.wrappers import Bidirectional, TimeDistributed
from mimic3models.keras_utils import LastTimestep
from mimic3models.keras_utils import ExtendMask
import tensorflow as tf
import numpy as np
import keras

class Network(Model):

    def __init__(self, dim, batch_norm, dropout, rec_dropout, task,
                 target_repl=False, deep_supervision=False, num_classes=1,
                 depth=1, input_dim=76, **kwargs):
        """args::
        task: 'decomp', 'ihm' or 'ph'
        """

        print("==> not used params in network class:", kwargs.keys())

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth

        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            if num_classes == 1:
                final_activation = 'relu'
            else:
                final_activation = 'softmax'
        else:
            raise ValueError("Wrong value for task")

        # Input layers and masking
        X = Input(shape=(None, input_dim), name='X')
        inputs = [X]
        mX = Masking()(X)

        if deep_supervision:
            M = Input(shape=(None,), name='M')
            inputs.append(M)

        # Configurations
        is_bidirectional = True
        if deep_supervision:
            is_bidirectional = False

        # Main part of the network
        for i in range(depth - 1):
            num_units = dim
            if is_bidirectional:
                num_units = num_units // 2

            lstm = LSTM(units=num_units,
                        activation='tanh',
                        return_sequences=True,
                        recurrent_dropout=rec_dropout,
                        dropout=dropout)

            if is_bidirectional:
                mX = Bidirectional(lstm)(mX)
            else:
                mX = lstm(mX)

        # Output module of the network
        return_sequences = (target_repl or deep_supervision)
        L = LSTM(units=dim,
                 activation='tanh',
                 return_sequences=return_sequences,
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(mX)

        if dropout > 0:
            L = Dropout(dropout)(L)

        if target_repl:
            y = TimeDistributed(Dense(num_classes, activation=final_activation),
                                name='seq')(L)
            y_last = LastTimestep(name='single')(y)
            outputs = [y_last, y]
        elif deep_supervision:
            y = TimeDistributed(Dense(num_classes, activation=final_activation))(L)
            y = ExtendMask()([y, M])  # this way we extend mask of y to M
            outputs = [y]
        else:
            y = Dense(num_classes, activation=final_activation)(L)
            outputs = [y]

        super(Network, self).__init__(inputs=inputs, outputs=outputs)

    def say_name(self):
        return "{}.n{}{}{}{}.dep{}".format('k_lstm',
                                           self.dim,
                                           ".bn" if self.batch_norm else "",
                                           ".d{}".format(self.dropout) if self.dropout > 0 else "",
                                           ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                                           self.depth)


class CPLSTM(tf.keras.Model):
    def __init__(self, dim, batch_norm, dropout, rec_dropout, task, predstep, supervise=False,
                 target_repl=False, is_bidirectional=True, num_classes=1,
                 depth=1, input_dim=76, **kwargs):
        super(CPLSTM, self).__init__()
        print("==> not used params in network class:", kwargs.keys())

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        self.predstep=predstep
        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            if num_classes == 1:
                final_activation = 'relu'
            else:
                final_activation = 'softmax'
        else:
            raise ValueError("Wrong value for task")




        # Configurations
        l=[Masking()]
        num_units = dim
        for i in range(depth - 1):
            if is_bidirectional:
                num_units = num_units // 2
                l.append(Bidirectional(LSTM(units=num_units,
                        activation='tanh',
                        return_sequences=True,
                        recurrent_dropout=rec_dropout,
                        dropout=dropout)))

            else:
                l.append(LSTM(units=num_units,
                        activation='tanh',
                        return_sequences=True,
                        recurrent_dropout=rec_dropout,
                        dropout=dropout))
        l.append(LSTM(units=dim,
                 activation='tanh',
                 return_sequences=True,
                          return_state=True,
                          dropout=dropout,
                 recurrent_dropout=rec_dropout)
)
        if dropout>0:l.append(Dropout(dropout))


        self.encodeRegress=keras.Sequential(layers=[*l])
        self.linears= [Dense(num_units,activation='tanh') for i in range(predstep+1)]
        self.softmax=keras.layers.Softmax(axis=0)


        def call(self, x, **kwargs):
            self.bs,self.duration,self.features=x.shape[0],x.shape[1],x.shape[2]
            # which time point to predict from
            t_samples=random.randint(int(self.duration/3),self.duration-self.predstep-1)
            _, zt, ct =self.encodeRegress(x[:,:t_samples,:])
            # zt is the last latent vector. ct is the last output
            z_embed= np.empty((self.bs,self.predstep,self.features))
            z_preds= np.empty((self.bs,self.predstep,self.features))

            for i in range(self.predstep):
                _,z_embed[:,i,:],_=self.encodeRegress(x[:,t_samples+i,:])
                z_preds[:,i,:]=self.linears[i](ct)

            # ----------  NCE loss ------------
            nce=0
            for i in np.arange(0,self.predstep):
                total=tf.matmul(z_embed[:,i,:],np.transpose(z_preds[:,i]))
                correct=tf.keras.backend.sum(tf.math.equal(tf.argmax(self.softmax(total),axis=0),
                                                           tf.range(0,self.bs)))
                nce+=tf.keras.backend.sum(tf.linalg.diag(tf.nn.log_softmax(total,axis=0)))
            nce /= -1 * self.bs * self.predstep
            accuracy = 1. * correct/self.bs

            return nce

