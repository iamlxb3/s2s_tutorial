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
                yield (x_batch, y_batch)
                x_batch = []
                y_batch = []


def lstm():
    #model = Sequential()
    # model.add(Embedding(10, 100, input_length=8))
    # model.add(LSTM(20, return_sequences=True))
    # model.add(LSTM(20, return_sequences=True))
    # model.add(TimeDistributed(Dense(10)))
    # model.add(Dense(10, activation='sigmoid'))
    # model.compile(loss='categorical_crossentropy', optimizer=adam(lr=1e-5))
    # model.summary()
    model = Sequential()
    model.add(Bidirectional(LSTM(128, input_shape=(8, 10), return_sequences=True),merge_mode='concat', weights=None))
    model.add(TimeDistributed(Dense(10, activation='softmax')))

    model.compile(loss='categorical_crossentropy', optimizer=adam(lr=1e-3))
    #model.add(Dense(10, activation='softmax'))
    #model.compile(loss='categorical_crossentropy', optimizer=adam(lr=1e-3))
    #print(model.summary())

    return model

def encode_decoder():

    encoder_dim = 20
    decoder_dim = 40

    encoder_inputs = Input(shape=(8, 10))

    encoder_forward = LSTM(encoder_dim, input_shape=(8, 10), return_state=True)
    _, state_h_f, state_c_f = encoder_forward(encoder_inputs)

    encoder_backward = LSTM(encoder_dim, input_shape=(8, 10), return_state=True, go_backwards=True)
    _, state_h_b, state_c_b = encoder_backward(encoder_inputs)


    # We discard `encoder_outputs` and only keep the states.
    state_h = merge.concatenate([state_h_f, state_h_b])
    state_c = merge.concatenate([state_c_f, state_c_b])

    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(8, 10))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(decoder_dim, input_shape=(8, 10), return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(10, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary() # maybe Keras bug?

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
    model = SimpleSeq2Seq(input_dim=10, hidden_dim=50, output_length=8, output_dim=10, depth=1)
    # model = AttentionSeq2Seq(input_dim=10, input_length=8, hidden_dim=250, output_length=8, output_dim=10, depth=1)

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
        X.append(np.array([np.load(x_npy)]))
        Y.append(np.array([np.load(y_npy)]))
    X = np.concatenate(X)
    Y = np.concatenate(Y)

    return X, Y



# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)


def main():
    # config
    batch_size = 10

    # load model
    model1 = model()
    #model1 = lstm()
    #model1 = encode_decoder()

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


    # get the encoder for train
    train_decoder_Y = train_Y.copy()
    for i, ty in enumerate(train_decoder_Y):
        new_ty = np.roll(ty ,1, axis=0)
        new_ty[0] *= 0.0
        train_decoder_Y[i] = new_ty

    # get the encoder for val
    val_decoder_Y = val_Y.copy()
    for i, ty in enumerate(val_decoder_Y):
        new_ty = np.roll(ty ,1, axis=0)
        new_ty[0] *= 0.0
        val_decoder_Y[i] = new_ty


    # train
    # model1.fit([train_X, train_decoder_Y],
    #            train_Y,
    #            batch_size=batch_size,
    #            validation_data=([val_X, val_decoder_Y], val_Y),
    #            epochs=400,
    #            )

    model1.fit(train_X,
               train_Y,
               batch_size=batch_size,
               validation_data=(val_X, val_Y),
               epochs=400,
               )



main()