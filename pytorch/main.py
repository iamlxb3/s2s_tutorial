# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

import config
from rnn_encoder import EncoderRNN
from attention_decoder import AttnDecoderRNN
from trainer import trainIters
import numpy as np
import torch
import random
import torch.nn as nn
import ipdb
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sentence_to_tensor(sentence):
    words = sentence.split(' ')
    input_length = len(words)
    array = np.random.random((input_length, 256))
    tensor = torch.from_numpy(array)
    tensor = tensor.float()
    return tensor, input_length


def sentence_to_int_tensors(sentence):
    int_tensors = []
    words = sentence.split(' ')
    for _ in words:
        array = np.array([random.randint(0, Vocab_len)])
        tensor = torch.from_numpy(array)
        int_tensors.append(tensor)
    return int_tensors


def evaluate(encoder, decoder, src_tensors, target_tensors, vocab, max_length=20, SOS_TOKEN_index=0):
    with torch.no_grad():

        input_length = len(src_tensors)

        # encoding
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(src_tensors[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        #

        # decoding
        decoder_input = torch.tensor([[SOS_TOKEN_index]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_outputs = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data

            # topv: tensor([[-7.4678]]), topi: tensor([[ 305]])
            # topv, topi = decoder_output.data.topk(1)
            decoder_output_t = decoder_output.data
            decoded_outputs.append(decoder_output_t)
            # if topi.item() == EOS_TOKEN:
            #     decoded_words.append('<EOS>')
            #     break
            # else:
            #     decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = decoder_output_t.topk(1)[1].squeeze().detach()
        #

        # compute loss
        criterion = nn.NLLLoss()
        loss = 0
        for i, gt_output in enumerate(target_tensors):
            decoded_output = decoded_outputs[i]
            loss += criterion(decoded_output, gt_output)
        loss = float(loss / len(target_tensors))
        print("loss: ", loss)
        #

        # decoded_outputs -> words
        decoded_words = []
        for decoder_output in decoded_outputs:
            _, topi = decoder_output.data.topk(1)
            word_index = int(topi[0][0])
            word = vocab[word_index]
            decoded_words.append(word)
            if word == 'EOS':
                break
        #
        print("Decoded_words: ", decoded_words)
        return decoded_words, decoder_attentions[:di + 1]


def torch_random_gen(X_paths, Y_paths):
    X_Y_paths = list(zip(X_paths, Y_paths))

    while True:
        x_path, y_path = random.choice(X_Y_paths)
        x = np.load(x_path)
        y = np.load(y_path)
        tensor_input = torch.from_numpy(x)
        tensor_input = tensor_input.float()
        tensor_output = torch.from_numpy(y)
        tensor_output = tensor_output.type(torch.LongTensor)

        yield (tensor_input, tensor_output)


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
    epoches = 1
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
    src_sentence = "I am big SB"
    target_sentence = "big SB am I"
    vocab = [x for x in range(2925)]

    # src
    src_tensors, input_length = sentence_to_tensor(src_sentence)
    target_tensors = sentence_to_int_tensors(target_sentence)

    decoded_words, attentions = evaluate(encoder1, attn_decoder1, src_tensors, target_tensors, vocab, max_length=10,
                                         SOS_TOKEN_index=0)
    #
