# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

import config
from rnn_encoder import EncoderRNN
from attention_decoder import AttnDecoderRNN
from trainer import trainIters
import numpy as np
import torch
import random
import pandas as pd
import torch.nn as nn
import ipdb
import glob
import os
import re
import pickle
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_emlo():
    from allennlp.modules.elmo import Elmo, batch_to_ids

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway" \
                   "/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway" \
                  "/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    return elmo, batch_to_ids


def sentence_to_tensor(sentence, elmo, batch_to_ids):
    words = word_tokenize(sentence, language='english')
    eng_ids = batch_to_ids([words])
    embeddings = elmo(eng_ids)
    tensor = embeddings['elmo_representations'][0][0]
    return tensor


def french_sentence_to_int_tensors(sentence, vocab):
    words = word_tokenize(sentence, language='french')

    tensor_output = np.zeros((len(words), 1))

    for i, word in enumerate(words):
        word_index = vocab.index(word)
        tensor_output[i] = word_index

    tensor_output = torch.from_numpy(tensor_output)
    tensor_output = tensor_output.type(torch.LongTensor)

    return tensor_output


def evaluate(encoder, decoder, src_tensors, target_tensors, vocab, max_length=20, SOS_token=0, EOS_token=0):
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
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_outputs = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data

            # topv: tensor([[-7.4678]]), topi: tensor([[ 305]])
            # topv, topi = decoder_output.data.topk(1)
            decoder_output_t = decoder_output.data


            #topv, topi = decoder_output_t.topk(1)
            # if topi.item() == EOS_token:
            #     break
            # else:
            #     decoded_outputs.append(decoder_output_t)

            # else:
            #     decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = decoder_output_t.topk(1)[1].squeeze().detach()
            decoded_outputs.append(decoder_output_t)
        #

        # compute loss
        criterion = nn.NLLLoss()
        loss = 0

        for i, gt_output in enumerate(target_tensors):
            decoded_output = decoded_outputs[i]
            loss += criterion(decoded_output, gt_output)
        loss = float(loss / len(target_tensors))

        #

        # decoded_outputs -> words
        decoded_words = []
        for decoder_output in decoded_outputs:
            _, topi = decoder_output.data.topk(1)
            word_index = int(topi[0][0])
            word = vocab[word_index]
            decoded_words.append(word)
            if word == '<EOS>':
                break
        #
        return loss, decoded_words, decoder_attentions[:di + 1]


def torch_random_train_gen(X_paths, Y_csv_path):
    # get y_dict
    y_df = pd.read_csv(Y_csv_path)
    y_dict = y_df.set_index('id').to_dict()['index']
    #

    while True:
        x_path = random.choice(X_paths)
        tensor_input = torch.load(x_path)
        id = int(re.findall(r'x_([0-9]+).pt', x_path)[0])
        y_indexes = y_dict[id].split(',')
        tensor_output = np.zeros((len(y_indexes), 1))
        for i, y_index in enumerate(y_indexes):
            tensor_output[i] = int(y_index)
        tensor_output = torch.from_numpy(tensor_output)
        tensor_output = tensor_output.type(torch.LongTensor)

        # tensor_input = torch.from_numpy(x)
        # tensor_input = tensor_input.float()
        # tensor_output = torch.from_numpy(y)
        # tensor_output = tensor_output.type(torch.LongTensor)

        yield (tensor_input, tensor_output)

def torch_random_val_gen(X_paths, Y_csv_path):
    # get y_dict
    y_df = pd.read_csv(Y_csv_path)
    y_dict = y_df.set_index('id').to_dict()['index']
    #

    for x_path in X_paths:
        tensor_input = torch.load(x_path)
        id = int(re.findall(r'x_([0-9]+).pt', x_path)[0])
        y_indexes = y_dict[id].split(',')
        tensor_output = np.zeros((len(y_indexes), 1))
        for i, y_index in enumerate(y_indexes):
            tensor_output[i] = int(y_index)
        tensor_output = torch.from_numpy(tensor_output)
        tensor_output = tensor_output.type(torch.LongTensor)

        # tensor_input = torch.from_numpy(x)
        # tensor_input = tensor_input.float()
        # tensor_output = torch.from_numpy(y)
        # tensor_output = tensor_output.type(torch.LongTensor)

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


if __name__ == '__main__':
    # model config
    hidden_size = 246
    encoder_nlayers = 1
    input_dim = 1024
    #

    # set path
    data_set = 'eng_fra'
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data')
    train_x_dir = os.path.join(data_dir, data_set, 'train')
    y_csv_path = os.path.join(data_dir, data_set, 'train_y.csv')
    vocab_path = os.path.join(data_dir, data_set, 'vocab.pkl')
    #

    # training config
    vocab = pickle.load(open(vocab_path, 'rb'))
    Vocab_len = len(vocab)
    EOS_token = int(vocab.index('<EOS>'))
    SOS_token = int(vocab.index('<SOS>'))

    ipdb.set_trace()
    N = 500
    epoches = 10
    batch_size = 1
    Y_max_length = 88
    #

    # create model
    encoder1 = EncoderRNN(input_dim, hidden_size, encoder_nlayers).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, Vocab_len, dropout_p=0.1, max_length=Y_max_length).to(device)
    print('model initialization ok!')
    #

    # get generator
    x_paths = glob.glob(os.path.join(train_x_dir, '*.pt'))[0:N]
    random.shuffle(x_paths)
    val_percent = 0.2
    val_index = int(N * 0.2)
    train_x_paths = x_paths[val_index:N]
    val_x_paths = x_paths[0:val_index]
    train_generator = torch_random_train_gen(train_x_paths, y_csv_path)
    val_generator = torch_random_val_gen(val_x_paths, y_csv_path)

    step_size = int(N / batch_size)
    #

    # start training
    trainIters(train_generator, encoder1, attn_decoder1, epoches, step_size, EOS_token, SOS_token, learning_rate=0.01,
               Y_max_length=Y_max_length)
    print('training ok!')
    #

    # eval
    val_loss = []
    for src_tensor, target_tensor in val_generator:
        loss, decoded_words, attentions = evaluate(encoder1, attn_decoder1, src_tensor, target_tensor, vocab,
                                                   max_length=Y_max_length, SOS_token=SOS_token, EOS_token=EOS_token)
        val_loss.append(loss)
        print("Decoded_words: ", decoded_words)

    print("val_loss: ", np.average(val_loss))
    #

    # # ------------------------------------------------------------------------------------------------------------------
    # # manual test
    # # ------------------------------------------------------------------------------------------------------------------
    # # init elmo
    # elmo, batch_to_ids = init_emlo()
    # from nltk.tokenize import word_tokenize
    # src_sentence = "I can't be sure."
    # target_sentence = "Je ne saurais en être certain."
    #
    # # src
    # src_tensors = sentence_to_tensor(src_sentence, elmo, batch_to_ids)
    # target_tensors = french_sentence_to_int_tensors(target_sentence, vocab)
    # # ------------------------------------------------------------------------------------------------------------------
