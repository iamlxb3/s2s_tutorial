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

    input_length = input_tensor.size(0)  # torch.Size([10, 1024])
    target_length = target_tensor.size(0)  # torch.Size([9, 1])

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
            # target_tensor_i = torch.from_numpy(np.array([np.argmax(target_tensor[di])]))
            # decoder_output: torch.Size([1, 30212])
            # target_tensor_i: tensor([ 30211])

            loss += criterion(decoder_output, target_tensor_i)
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def new_train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
              SOS_token, EOS_token, max_length, use_teacher_forcing=False):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    batch_size = input_tensor.size(0)
    target_length = target_tensor.size(1)  # torch.Size([9, 1])

    loss = 0

    # encoder_outputs : torch.Size([time_steps, batch_size, 256]),
    # encoder_hidden: torch.Size([num_layers * num_directions, batch_size, 256])

    encoder_h0 = encoder.initHidden(batch_size)
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_h0)

    decoder_input = target_tensor.view(target_tensor.shape[1], target_tensor.shape[0], -1)

    # this is teacher forcing
    # decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, encoder_hidden, encoder_outputs)


    # TODO, add teacher forcing
    # Without teacher forcing: use its own predictions as the next input
    for t in range(target_length):

        if t == 0:
            decoder_hidden = encoder_hidden
            decoder_input_t = torch.ones_like(decoder_input[t]) * SOS_token

        # input
        # decoder_input[t] : torch.Size([batch_size, 1])
        # decoder_hidden : torch.Size([layer*direction_num, batch_size, feature_dim])
        # encoder_outputs : torch.Size([decoder_length, batch_size, feature_dim])

        # output
        # decoder_output : torch.Size([1, batch_size, Vocab_size])
        # decoder_attention : torch.Size([1, batch_size, encoder_length])

        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input_t, decoder_hidden, encoder_outputs)

        decoder_output = decoder_output.squeeze(0)

        target_tensor_t = target_tensor[:, t, :]

        loss += criterion(decoder_output, target_tensor_t.squeeze(1))

        if use_teacher_forcing:
            _, topi = decoder_output.topk(1)
            decoder_input_t = topi.squeeze(0).detach()  # detach from history as input
        else:
            decoder_input_t = decoder_input[t-1]

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length # TODO


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

            loss = new_train(input_tensor, target_tensor, encoder,
                             decoder, encoder_optimizer, decoder_optimizer, criterion, SOS_token, EOS_token, max_length)
            epoch_loss += loss

            if verbose:
                print("Epoch-{} batch_index-{}/{} Loss: {}".format(epoch, batch_index, len(train_generator), loss))

        epoch_loss = epoch_loss / step_size
        print("Epoch: {}, loss: {}".format(epoch, epoch_loss))
