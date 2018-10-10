# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import random
import time
import math
import ipdb
import torch
import sys
import numpy as np
import torch.nn as nn
from torch import optim

# MAX_LENGTH = config.get_max_len()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
#           SOS_token, EOS_token, max_length, use_teacher_forcing=False, device=None):
#     encoder_hidden = encoder.initHidden()
#
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#
#     if len(input_tensor.shape) >= 3:
#         input_tensor = input_tensor.squeeze(0)
#         target_tensor = target_tensor.squeeze(0)
#
#     input_length = input_tensor.size(0)  # torch.Size([10, 1024])
#     target_length = target_tensor.size(0)  # torch.Size([9, 1])
#
#     encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#
#     loss = 0
#
#     for ei in range(input_length):
#         encoder_output, encoder_hidden = encoder(
#             input_tensor[ei], encoder_hidden)
#         encoder_outputs[ei] = encoder_output[0, 0]
#
#     decoder_input = torch.tensor([[SOS_token]], device=device)
#
#     decoder_hidden = encoder_hidden
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
#             # target_tensor_i = torch.from_numpy(np.array([np.argmax(target_tensor[di])]))
#             # decoder_output: torch.Size([1, 30212])
#             # target_tensor_i: tensor([ 30211])
#
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


def new_train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
              SOS_token, use_teacher_forcing=False, verbose=True, ignore_index=None,
              EOS_index=None, device=None):
    """
    Train for each batch

    """

    # initialize
    batch_size = input_tensor.shape[0]
    target_length = target_tensor.size(1)  # torch.Size([9, 1])
    loss = 0
    encoder_h0 = encoder.initHidden(batch_size, device)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    #

    # get the length of sequence for each sample
    seq_lens = []
    indices = []
    for batch_i in range(batch_size):
        seq_len = [float(sum(x)) for x in input_tensor[batch_i]]
        seq_len = len([x for x in seq_len if x != 0.0])
        seq_lens.append(seq_len)
        indices.append(batch_i)
    #


    # sort by decreasing order
    input_tensor_len = sorted(list(zip(input_tensor, seq_lens, indices)), key=lambda x:x[1], reverse=True)
    input_tensor = torch.cat([x[0].view(1, x[0].size(0), -1) for x in input_tensor_len], 0)
    sorted_seq_lens = [x[1] for x in input_tensor_len]
    sorted_indices = [x[2] for x in input_tensor_len]
    #


    # pack padded sequences
    input_tensor = torch.transpose(input_tensor, 0, 1)
    input_tensor = nn.utils.rnn.pack_padded_sequence(input_tensor, lengths=sorted_seq_lens)
    #


    # input_tensor: torch.Size([batch_size, time_steps, feature_dim])
    # encoder_h0: torch.Size([num_layers * num_directions, batch_size, 256])
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_h0)
    encoder_outputs = nn.utils.rnn.pad_packed_sequence(encoder_outputs)[0] # 0 -> tensor, 1 ->
    # length tensor([ 13,  11,  11,   8,   6,   6,   5,   4])
    encoder_outputs = encoder_outputs.index_select(1, torch.tensor(sorted_indices).to(device)) # recover indices
    # encoder_outputs : torch.Size([time_steps, batch_size, 256]),
    # encoder_hidden: torch.Size([num_layers * num_directions, batch_size, 256])

    # # padding after encoding TODO, modify in the future
    # new = torch.zeros((target_length, encoder_outputs.size(1), encoder_outputs.size(2)))
    # new[:encoder_outputs.size(0)] = encoder_outputs
    # encoder_outputs = new
    # #

    if verbose:
        decoded_outputs = []

    for t in range(target_length):

        if t == 0:
            decoder_hidden = encoder_hidden.to(device)
            decoder_input_t = (torch.ones((target_tensor.size(0), 1)) * SOS_token).long().to(device)

        # input
        # decoder_input_t : torch.Size([batch_size, 1])
        # decoder_hidden : torch.Size([layer*direction_num, batch_size, feature_dim])
        # encoder_outputs : torch.Size([decoder_length, batch_size, feature_dim])

        # output
        # decoder_output : torch.Size([1, batch_size, Vocab_size])
        # decoder_attention : torch.Size([1, batch_size, encoder_length])

        decoder_output, decoder_hidden, decoder_attention, context_vector\
            = decoder(decoder_input_t, decoder_hidden, encoder_outputs)

        decoder_output = decoder_output.squeeze(0)

        if verbose:
            decoded_output_t = decoder_output.topk(1)[1][0]
            #print("t: {}, decoder_output: {}".format(t, decoded_output_t))
            decoded_outputs.append(int(decoded_output_t))

        target_tensor_t = target_tensor[:, t, :]
        #print("t {}: target_tensor_t: {}".format(t, target_tensor_t))

        # TODO, add attention
        loss += criterion(decoder_output, target_tensor_t.squeeze(1))

        if not use_teacher_forcing:
            _, topi = decoder_output.topk(1)
            decoder_input_t = topi.squeeze(0).detach()  # detach from history as input
        else:
            decoder_input_t = target_tensor_t

        if t <= 8:
            # ipdb.set_trace()
            # decoder_attention_print = np.array(decoder_attention.detach())[0].reshape(decoder_attention.size(1))
            # print("decoder_attention: ", decoder_attention_print)
            # print("context_vector: {}, hidden: {}".format(torch.sum(context_vector), torch.sum(decoder_hidden)))
            print("t-{}, max-attention: {}".format(t, np.argmax(decoder_attention.detach(), axis=1)[0]))


    if verbose:
        print_target = [int(x) for x in target_tensor[0] if int(x) < int(ignore_index)]

        new_decoded_outputs = []
        for x in decoded_outputs:
            new_decoded_outputs.append(x)
            if x == EOS_index:
                break

        print("\n--------------------------------------------")
        print("decoded_outputs: ", new_decoded_outputs)
        print("target_tensor: ", print_target)
        print("Overlap: ", len(set(print_target).intersection(new_decoded_outputs)) / len(print_target))

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length  # TODO


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


def new_trainIters(train_generator, encoder, decoder, epoches, step_size, EOS_token, SOS_token, ignore_index,
                   learning_rate=0.01, verbose=False, use_teacher_forcing=False, device=None):
    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

    criterion = nn.NLLLoss(ignore_index=ignore_index)
    epoch_losses = []
    for epoch in range(epoches):
        epoch_loss = 0
        for batch_index, (input_tensor, target_tensor) in enumerate(train_generator):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            loss = new_train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                             criterion, SOS_token, use_teacher_forcing=use_teacher_forcing,
                             EOS_index=EOS_token, ignore_index=ignore_index, device=device)
            epoch_loss += loss

            if verbose:
                print("Epoch-{} batch_index-{}/{} Loss: {}".format(epoch, batch_index, len(train_generator), loss))

        epoch_loss = epoch_loss / step_size
        epoch_losses.append(epoch_loss)
        print("Epoch: {}, loss: {}".format(epoch, epoch_loss))

        print("epoch_losses: ", epoch_losses)