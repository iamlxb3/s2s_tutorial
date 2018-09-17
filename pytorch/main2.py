# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

import config
from rnn_encoder import EncoderRNN
from attention_decoder import AttnDecoderRNN
from trainer import trainIters
import numpy as np
import torch
import random
import ipdb
import sys


def train_gen():
    training_pairs = []
    N = 50
    for i in range(N):
        arr_input = np.random.random((5, 256))
        arr_output = np.zeros((5, 2925))
        for t_arr in arr_output:
            t_arr[random.randint(0, 2925)] = 1

        # np.array input -> torch
        tensor_input = torch.from_numpy(arr_input)
        tensor_input = tensor_input.float()

        # np.array output -> torch
        tensor_output = torch.from_numpy(arr_output)
        tensor_output = tensor_output.type(torch.LongTensor)

        training_pairs.append((tensor_input, tensor_output))

    while True:
        for training_pair in training_pairs:
            yield training_pair


device = config.get_device()

if __name__ == '__main__':
    # model config
    hidden_size = 256
    encoder_nlayers = 1
    input_dim = 256
    input_N = 50
    #

    # training config
    Vocab_len = 2925
    epoches = 10
    batch_size = 1
    #

    # create model
    encoder1 = EncoderRNN(input_dim, hidden_size, encoder_nlayers).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, Vocab_len, dropout_p=0.1).to(device)
    print('model initialization ok!')
    #

    # get generator
    train_generator = train_gen()
    step_size = int(input_N / batch_size)
    #

    # start training
    trainIters(train_generator, encoder1, attn_decoder1, epoches, step_size, learning_rate=0.01)
    print('training ok!')
    #

    # predict

    #
