# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import random
import time
import math
import ipdb
import torch
import numpy as np
import torch.nn as nn
from torch import optim

# MAX_LENGTH = config.get_max_len()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def batch_train(input_tensor_batch, target_tensor_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
#           SOS_token, EOS_token, max_length, batch_size, use_teacher_forcing=False):
#     encoder_hidden = encoder.initHidden()
#
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#
#     encoder_outputs = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)
#
#     loss = 0
#
#     for batch_index in range(batch_size):
#
#         input_tensor = input_tensor_batch[batch_index]
#         target_tensor = target_tensor_batch[batch_index]
#
#         input_length = input_tensor.size(0)
#         target_length = target_tensor.size(0)
#
#         for ei in range(input_length):
#             encoder_output, encoder_hidden = encoder(
#                 input_tensor[ei], encoder_hidden)
#             encoder_outputs[batch_index][ei] = encoder_output[0, 0]
#
#     decoder_input = torch.tensor([[SOS_token]], device=device)
#
#     decoder_hidden = encoder_hidden
#
#
#     if use_teacher_forcing:
#         # Teacher forcing: Feed the target as the next input
#         for di in range(target_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             loss += criterion(decoder_output, target_tensor[di])
#             decoder_input = target_tensor[di]  # Teacher forcing
#
#     else:
#         # Without teacher forcing: use its own predictions as the next input
#         for di in range(target_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             topv, topi = decoder_output.topk(1)
#             decoder_input = topi.squeeze().detach()  # detach from history as input
#             target_tensor_i = target_tensor[di]
#             #target_tensor_i = torch.from_numpy(np.array([np.argmax(target_tensor[di])]))
#             loss += criterion(decoder_output, target_tensor_i)
#             if decoder_input.item() == EOS_token:
#                 break
#
#     loss.backward()
#
#     encoder_optimizer.step()
#     decoder_optimizer.step()
#
#     return loss.item() / target_length

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          SOS_token, EOS_token, max_length, use_teacher_forcing=False):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()


    if len(input_tensor.shape) >= 3:
        input_tensor = input_tensor.squeeze(0)
        target_tensor = target_tensor.squeeze(0)

    input_length = input_tensor.size(0) # torch.Size([10, 1024])
    target_length = target_tensor.size(0) # torch.Size([9, 1])


    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden


    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            target_tensor_i = target_tensor[di]
            #target_tensor_i = torch.from_numpy(np.array([np.argmax(target_tensor[di])]))
            # decoder_output: torch.Size([1, 30212])
            # target_tensor_i: tensor([ 30211])

            loss += criterion(decoder_output, target_tensor_i)
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
#

def trainIters(train_generator, encoder, decoder, epoches, step_size, EOS_token, SOS_token,
               learning_rate=0.01, max_length=None, verbose=False):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for epoch in range(epoches):
        epoch_loss = 0
        for step in range(step_size):

            training_pair = next(train_generator)  # TODO, modify if the batchsize is not 1!
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, SOS_token, EOS_token, max_length)
            epoch_loss += loss

            if verbose:
                print("Epoch-{} Step-{} Loss: {}".format(epoch, step, loss))

        epoch_loss = epoch_loss / step_size
        print("Epoch: {}, loss: {}".format(epoch, epoch_loss))

def new_trainIters(train_generator, encoder, decoder, epoches, step_size, EOS_token, SOS_token,
               learning_rate=0.01, max_length=None, verbose=False):


    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

    criterion = nn.NLLLoss()

    for epoch in range(epoches):
        epoch_loss = 0
        for batch_index, (input_tensor, target_tensor) in enumerate(train_generator):

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, SOS_token, EOS_token, max_length)
            epoch_loss += loss

            if verbose:
                print("Epoch-{} batch_index-{} Loss: {}".format(epoch, batch_index, loss))

        epoch_loss = epoch_loss / step_size
        print("Epoch: {}, loss: {}".format(epoch, epoch_loss))