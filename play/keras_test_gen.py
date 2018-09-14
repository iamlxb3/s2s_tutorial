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
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, adam
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Embedding, TimeDistributed
from seq2seq.models import AttentionSeq2Seq

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
                yield (x_batch, y_batch)
                x_batch = []
                y_batch = []

def lstm():
    model = Sequential()
    model.add(Embedding(10, 1000, input_length=8))
    model.add(LSTM(20, return_sequences=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(TimeDistributed(Dense(10)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=adam(lr=1e-5))
    model.summary()
    return model


# def model():
#     model = Sequential()
#     model.add(Dense(16, activation='relu', input_shape=(10,)))
#     model.add(Dropout(0.2))
#     model.add(Dense(output_dim=2, activation='softmax'))
#     model.summary()
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=RMSprop(lr=0.00001),
#                   metrics=['accuracy'])


def model():
    model = SimpleSeq2Seq(input_dim=10, hidden_dim=250, output_length=8, output_dim=10, depth=1)
    #model = AttentionSeq2Seq(input_dim=10, input_length=8, hidden_dim=250, output_length=8, output_dim=10, depth=1)

    model.compile(loss='categorical_crossentropy', optimizer=adam(lr=1e-4))
    #model.compile(loss='mse', optimizer=adam(lr=1e-3))

    model.summary()
    return model


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


def main():
    # config
    batch_size = 10

    # load model
    model1 = model()
    #model1 = lstm()

    # load data
    data_dir = '/Users/pjs/byte_play/data/toy_data2/train'
    X, Y = train_X_Y_load(data_dir)
    print("Load train paths complete!")

    # split into train/val
    train_X = X[:120]
    train_Y = Y[:120]
    val_X = X[120:]
    val_Y = Y[120:]
    train_N = len(train_X)
    val_N = len(val_X)

    train_gen = random_gen(train_X, train_Y, batch_size=batch_size)
    train_step_size = int(math.ceil(train_N / batch_size))
    val_gen = random_gen(val_X, val_Y, batch_size=batch_size)
    val_step_size = int(math.ceil(val_N / batch_size))

    # train
    model1.fit_generator(generator=train_gen,
                         epochs=300,
                         validation_data=val_gen,
                         steps_per_epoch=train_step_size,
                         validation_steps=val_step_size,
                         )


main()
