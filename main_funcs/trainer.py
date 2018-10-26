# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import collections
import ipdb
import torch
import sys
import numpy as np
import torch.nn as nn
from torch import optim


def train_1_batch(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                  SOS_token, use_teacher_forcing=False, verbose=True, ignore_index=None,
                  EOS_index=None, device=None, is_record_attention=False):
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
    input_tensor_len = sorted(list(zip(input_tensor, seq_lens, indices)), key=lambda x: x[1], reverse=True)
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
    encoder_outputs = nn.utils.rnn.pad_packed_sequence(encoder_outputs)[0]  # 0 -> tensor, 1 ->
    # length tensor([ 13,  11,  11,   8,   6,   6,   5,   4])
    encoder_outputs = encoder_outputs.index_select(1, torch.tensor(sorted_indices).to(device))  # recover indices
    # encoder_outputs : torch.Size([time_steps, batch_size, 256]),
    # encoder_hidden: torch.Size([num_layers * num_directions, batch_size, 256])

    # # padding after encoding TODO, modify in the future
    # new = torch.zeros((target_length, encoder_outputs.size(1), encoder_outputs.size(2)))
    # new[:encoder_outputs.size(0)] = encoder_outputs
    # encoder_outputs = new
    # #

    if verbose:
        decoded_outputs = []

    # add attention recoder
    attention_recoder = collections.defaultdict(lambda: [])
    #

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

        decoder_output, decoder_hidden, decoder_attention, context_vector \
            = decoder(decoder_input_t, decoder_hidden, encoder_outputs)

        decoder_output = decoder_output.squeeze(0)

        if verbose:
            decoded_output_t = decoder_output.topk(1)[1][0]
            # print("t: {}, decoder_output: {}".format(t, decoded_output_t))
            decoded_outputs.append(int(decoded_output_t))

        target_tensor_t = target_tensor[:, t, :]
        # print("t {}: target_tensor_t: {}".format(t, target_tensor_t))

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

        if is_record_attention:
            # record attention
            attention_pos_on_t = list(np.argmax(decoder_attention.detach(), axis=1).squeeze())
            attention_pos_on_t = [int(x) for x in attention_pos_on_t]
            attention_recoder[t].extend(attention_pos_on_t)
            #

    samples_attention = None
    if is_record_attention:
        # get the batch length
        samples_attention = []
        for batch in range(batch_size):
            batch_length = target_tensor[batch][target_tensor[batch] < ignore_index].size(0)
            sample_attentions = np.zeros(batch_length)
            for t in range(batch_length):
                sample_attentions[t] = attention_recoder[t][batch]
            samples_attention.append(sample_attentions)
        #

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

    return loss.item() / target_length, samples_attention  # TODO


def epoches_train(train_generator, encoder, decoder, epoches, step_size, EOS_token, SOS_token, ignore_index,
                  learning_rate=0.01, verbose=False, use_teacher_forcing=False, device=None):
    # set optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

    # set criterion
    criterion = nn.NLLLoss(ignore_index=ignore_index)

    epoch_losses = []  # TODO, temp track loss

    # add attention-recoder
    for epoch in range(epoches):

        if epoch == epoches - 1:
            is_record_attention = True
        else:
            is_record_attention = False

        epoch_loss = 0
        attention_recoder = []
        for batch_index, (input_tensor, target_tensor, uid) in enumerate(train_generator):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            loss, samples_attention = train_1_batch(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
                                                    decoder_optimizer,
                                                    criterion, SOS_token, use_teacher_forcing=use_teacher_forcing,
                                                    EOS_index=EOS_token, ignore_index=ignore_index, device=device,
                                                    is_record_attention=is_record_attention)

            if samples_attention:
                attention_recoder.extend(list(zip([int(x) for x in uid], samples_attention)))

            epoch_loss += loss

            if verbose:
                print("Epoch-{} batch_index-{}/{} Loss: {}".format(epoch, batch_index, len(train_generator), loss))

        epoch_loss = epoch_loss / step_size
        epoch_losses.append(epoch_loss)
        print("Epoch: {}, loss: {}".format(epoch, epoch_loss))

        print("epoch_losses: ", epoch_losses)

        # TODO, analysis attention at the last epoch
        print(attention_recoder)
        #


def epoches_train2(config, train_generator, encoder, decoder):
    # device
    device = config.device

    # set optimizer
    encoder_optimizer = config.encoder_optimizer
    decoder_optimizer = config.decoder_optimizer

    # set criterion
    criterion = config.criterion

    epoch_losses = []  # TODO, temp track loss

    # add attention-recoder
    for epoch in range(config.epoches):

        epoch_loss = 0
        for batch_index, (input_tensor, target_tensor, uid) in enumerate(train_generator):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            loss = train_1_batch_basic_rnn(input_tensor, target_tensor, encoder, decoder,
                                                              encoder_optimizer,
                                                              decoder_optimizer,
                                                              criterion,
                                                              use_teacher_forcing=config.use_teacher_forcing,
                                                              src_pad_token=config.src_pad_token,
                                                              target_SOS_token=config.target_SOS_token,
                                                              target_EOS_index=config.target_EOS_token,
                                                              target_pad_token=config.target_pad_token, device=device)

            epoch_loss += loss

            if config.verbose:
                print("Epoch-{} batch_index-{}/{} Loss: {}".format(epoch, batch_index, len(train_generator), loss))

        epoch_loss = epoch_loss / config.step_size
        epoch_losses.append(epoch_loss)
        print("Epoch: {}, loss: {}".format(epoch, epoch_loss))
        print("epoch_losses: ", epoch_losses)


def _actual_seq_length_compute(input_tensor, batch_size, src_pad_token):
    seq_lens = []
    indices = []
    for batch_i in range(batch_size):
        seq_len = input_tensor[0].squeeze(1)
        seq_len = len([x for x in seq_len if x != src_pad_token])
        seq_lens.append(seq_len)
        indices.append(batch_i)
    #
    return seq_lens, indices

def _sort_batch_seq(input_tensor, batch_size, src_pad_token):

    # get the actual length of sequence for each sample
    src_seq_lens, src_seq_indices = _actual_seq_length_compute(input_tensor, batch_size, src_pad_token)
    #

    # sort by decreasing order
    input_tensor_len = sorted(list(zip(input_tensor, src_seq_lens, src_seq_indices)), key=lambda x: x[1], reverse=True)
    input_tensor = torch.cat([x[0].view(1, x[0].size(0), -1) for x in input_tensor_len], 0)
    sorted_seq_lens = [x[1] for x in input_tensor_len]
    sorted_indices = [x[2] for x in input_tensor_len]
    #
    return input_tensor, sorted_seq_lens, sorted_indices

def train_1_batch_basic_rnn(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                            criterion,
                            verbose=True,
                            device=None,
                            use_teacher_forcing=False,
                            target_pad_token=None,
                            src_pad_token=None,
                            target_SOS_token=None,
                            target_EOS_index=None):
    """

    :param input_tensor: torch.Size([batch_size, max_padding_len, 1])
    :param target_tensor: torch.Size([batch_size, max_padding_len, 1])
    :param target_pad_token: int
    :param target_SOS_token: int
    :param target_EOS_index: int
    """

    # initialize
    batch_size = input_tensor.shape[0]
    target_length = target_tensor.size(1)  # torch.Size([9, 1])
    loss = 0
    encoder_h0 = encoder.initHidden(batch_size, device)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    #

    # get the actual length of sequence for each sample, sort by decreasing order
    input_tensor, sorted_seq_lens, sorted_indices = _sort_batch_seq(input_tensor, batch_size, src_pad_token)
    input_tensor = torch.transpose(input_tensor, 0, 1) # transpose, batch second
    #

    # encode
    encoder_outputs, encoder_hidden = encoder(input_tensor, sorted_seq_lens, sorted_indices, encoder_h0)
    #

    if verbose:
        decoded_outputs = []

    for t in range(target_length):

        if t == 0:
            decoder_hidden = encoder_hidden
            decoder_input_t = (torch.ones((target_tensor.size(0), 1)) * target_SOS_token).long().to(device)

        # input
        # decoder_input_t : torch.Size([batch_size, 1])
        # decoder_hidden : torch.Size([layer*direction_num, batch_size, feature_dim])
        # encoder_outputs : torch.Size([decoder_length, batch_size, feature_dim])

        # output
        # decoder_output : torch.Size([1, batch_size, Vocab_size])

        decoder_output, decoder_hidden = decoder(decoder_input_t, decoder_hidden)

        decoder_output = decoder_output.squeeze(0)

        if verbose:
            decoded_output_t = decoder_output.topk(1)[1][0]
            # print("t: {}, decoder_output: {}".format(t, decoded_output_t))
            decoded_outputs.append(int(decoded_output_t))

        target_tensor_t = target_tensor[:, t, :]
        # print("t {}: target_tensor_t: {}".format(t, target_tensor_t))

        # TODO, add attention
        loss += criterion(decoder_output, target_tensor_t.squeeze(1))

        if not use_teacher_forcing:
            _, topi = decoder_output.topk(1)
            decoder_input_t = topi.squeeze(0).detach()  # detach from history as input
        else:
            decoder_input_t = target_tensor_t


    # if verbose:
    #     print_target = [int(x) for x in target_tensor[0] if int(x) != target_pad_token]
    #     new_decoded_outputs = []
    #     for x in decoded_outputs:
    #         new_decoded_outputs.append(x)
    #         if x == target_EOS_index:
    #             break
    #
    #     print("\n--------------------------------------------")
    #     print("decoded_outputs: ", new_decoded_outputs)
    #     print("target_tensor: ", print_target)
    #     print("Overlap: ", len(set(print_target).intersection(new_decoded_outputs)) / len(print_target))

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
