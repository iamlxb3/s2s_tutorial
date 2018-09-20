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


# SOS_TOKEN = config.get_sos_token()
# EOS_TOKEN = config.get_eos_token()
# teacher_forcing_ratio = config.get_teacher_forcing_ratio()


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          SOS_token, EOS_token, max_length, use_teacher_forcing=False):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)


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
            loss += criterion(decoder_output, target_tensor_i)
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(train_generator, encoder, decoder, epoches, step_size, EOS_token, SOS_token,
               learning_rate=0.01, max_length=None):
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
        epoch_loss = epoch_loss / step_size
        print("Epoch: {}, loss: {}".format(epoch, epoch_loss))
