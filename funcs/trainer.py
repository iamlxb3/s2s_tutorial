# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import sys
import torch
import numpy as np

sys.path.append('..')
from utils.helpers import _sort_batch_seq
from utils.helpers import save_cktpoint
from funcs.eval_predict import eval_on_val

def epoches_train(config, train_loader, val_loader, encoder, decoder, epoch_recorder, encoder_path, decoder_path):
    # device
    device = config.device

    # set criterion
    criterion = config.criterion

    # add attention-recoder
    for epoch, epoch_index in enumerate(range(config.epoches)):

        epoch_loss = 0
        for batch_index, (batch_x, batch_y, uid) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            loss = train_1_batch_basic_rnn(batch_x, batch_y, encoder, decoder,
                                           config.optimizer,
                                           criterion,
                                           use_teacher_forcing=config.use_teacher_forcing,
                                           src_pad_token=config.src_pad_token,
                                           target_SOS_token=config.target_SOS_token,
                                           target_EOS_index=config.target_EOS_token,
                                           target_pad_token=config.target_pad_token, device=device)

            epoch_loss += loss

            if config.verbose:
                print("Epoch-{} batch_index-{}/{} Loss: {}".format(epoch, batch_index, len(train_loader), loss))

        # eval on validation set
        val_loss = []
        for X_val, Y_val, uid in val_loader:
            # evaluate on validation set
            loss = eval_on_val(encoder, decoder, X_val, Y_val, device, config.target_SOS_token, config.target_pad_token,
                               config.src_pad_token, teacher_forcing=False, batch_size=config.batch_size)
            val_loss.append(loss)
        val_loss = np.average(val_loss)
        config.lr_scheduler.step(val_loss)
        epoch_loss = epoch_loss / config.step_size
        #

        print("Epoch: {}, loss: {}, val_loss: {}".format(epoch, epoch_loss, val_loss))

        # Save checkpoint
        lowest_val_loss, lowest_val_loss_index = epoch_recorder.lowest_val_loss
        if val_loss < lowest_val_loss:
            save_cktpoint(encoder, decoder, encoder_path, decoder_path)
            print("val_loss: {}, epoch-{}, Save checkpoint to {}, {}".format(val_loss, epoch_index, encoder_path,
                                                                            decoder_path))
        else:
            print("val_loss no improvement, lowest: {}, epoch-{}".format(lowest_val_loss, lowest_val_loss_index))
        epoch_recorder.val_loss_update(val_loss)
        epoch_recorder.train_loss_update(epoch_loss)
        #



def train_1_batch_basic_rnn(input_tensor, target_tensor, encoder, decoder, optimizer,
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
    target_max_len = target_tensor.size(1)  # torch.Size([9, 1])
    loss = 0
    encoder_h0 = encoder.initHidden(batch_size, device)
    optimizer.zero_grad()
    #

    # get the actual length of sequence for each sample, sort by decreasing order
    input_tensor, sorted_seq_lens, sorted_indices = _sort_batch_seq(input_tensor, batch_size, src_pad_token)
    input_tensor = torch.transpose(input_tensor, 0, 1)  # transpose, batch second
    #

    # encode
    encoder_outputs, encoder_hidden = encoder(input_tensor, sorted_seq_lens, sorted_indices, encoder_h0)
    #

    if verbose:
        decoded_outputs = []

    # TODO, add teacher forcing ratio

    for t in range(target_max_len):

        if t == 0:
            decoder_hidden = encoder_hidden
            decoder_input_t = (torch.ones((target_tensor.size(0), 1)) * target_SOS_token).long().to(device)

        decoder_output, decoder_hidden = decoder(decoder_input_t, decoder_hidden)
        decoder_output = decoder_output.squeeze(0)

        if verbose:
            decoded_output_t = decoder_output.topk(1)[1][0]
            # print("t: {}, decoder_output: {}".format(t, decoded_output_t))
            decoded_outputs.append(int(decoded_output_t))

        target_tensor_t = target_tensor[:, t, :]

        if not use_teacher_forcing:
            topv, topi = decoder_output.topk(1)
            decoder_input_t = topi  # .detach() or not?  # detach from history as input
        else:
            decoder_input_t = target_tensor_t

        loss += criterion(decoder_output, target_tensor_t.squeeze(1))

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

    # calculate gradient & update parameters

    loss.backward()
    optimizer.step()

    avg_batch_loss = loss.item() / target_max_len
    return avg_batch_loss
