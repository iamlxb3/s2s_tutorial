"""
# https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
# https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
# https://gist.github.com/mbollmann/ccc735366221e4dba9f89d2aab86da1e # attention LSTM
# https://github.com/keras-team/keras/issues/1472
"""

import numpy as np
import keras
import math
import random
import glob
import os
import re
import ipdb
from keras.models import Sequential
from seq2seq.models import SimpleSeq2Seq
from keras.layers import Dense, Dropout, Bidirectional
from keras.optimizers import RMSprop, adam
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Embedding, TimeDistributed
from seq2seq.models import AttentionSeq2Seq
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers import add

from keras.layers import merge


def random_gen(X_paths, Y_paths, batch_size=10):
    while True:
        X_Y_paths = list(zip(X_paths, Y_paths))
        random.shuffle((X_Y_paths))
        X_paths, Y_paths = zip(*X_Y_paths)

        x_batch = []
        y_batch = []

        for i, x_path in enumerate(X_paths):
            y_path = Y_paths[i]
            x = np.load(x_path)
            y = np.load(y_path)
            x = np.array([x])
            y = np.array([y])
            x_batch.append(x)
            y_batch.append(y)

            assert len(x_batch) == len(y_batch)
            if len(x_batch) == batch_size:
                x_batch = np.concatenate(x_batch)
                y_batch = np.concatenate(y_batch)

                yield x_batch, y_batch
                x_batch = []
                y_batch = []


def train_X_Y_load(data_dir):
    x_npys = glob.glob(os.path.join(data_dir, 'x_*.npy'))
    y_npys = glob.glob(os.path.join(data_dir, 'y_*.npy'))
    assert len(x_npys) == len(y_npys)

    X = []
    Y = []
    for x_npy in x_npys:
        id = re.findall(r'x_([0-9]+).npy', x_npy)[0]
        y_npy = os.path.join(data_dir, 'y_{}.npy'.format(id))
        X.append(x_npy)
        Y.append(y_npy)
    return X, Y


def model():
    model = SimpleSeq2Seq(input_shape=(8, 10), hidden_dim=500, output_length=8, output_dim=10, depth=1)
    # model = AttentionSeq2Seq(input_dim=10, input_length=8, hidden_dim=250, output_length=8, output_dim=10, depth=1)
    model.compile(loss='categorical_crossentropy', optimizer=adam(lr=1e-4))
    # model.compile(loss='mse', optimizer=adam(lr=1e-3))
    model.summary()
    return model


# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


def main():
    # config
    batch_size = 10

    # load model
    model1 = model()

    # load data
    train_N = 100
    train_X = np.random.random((train_N, 8, 10))
    # train_Y = np.random.randint(8, size=(train_N, 8, 1))
    train_Y = train_X.copy()

    # train
    model1.fit(x=train_X,
               y=train_Y,
               batch_size=10,
               epochs=100,
               )


main()
