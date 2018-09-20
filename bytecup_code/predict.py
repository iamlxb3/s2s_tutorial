import numpy as np
import torch
import random
import ipdb
import glob
import os
import pickle
import sys
import re

sys.path.append("..")
from main_funcs.gen import torch_test_gen
from main_funcs.eval_predict import predict
from main_funcs.helpers import sort_test_xpaths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # model config
    hidden_size = 246
    encoder_nlayers = 1
    input_dim = 1024
    batch_size = 1
    Y_max_length = 1000
    #

    # set path
    data_set = 'bytecup2018'
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data')
    pkl_dir = os.path.join(top_dir, 'model_pkls')
    test_dir = os.path.join(data_dir, data_set, 'test') #TODO, modify
    vocab_path = os.path.join(data_dir, data_set, 'bytecup_vocab.pkl')
    encoder_path = os.path.join(pkl_dir, 'encoder.pkl')
    decoder_path = os.path.join(pkl_dir, 'decoder.pkl')
    save_dir = os.path.join(data_dir, data_set, 'result')
    #

    # create model
    encoder1 = torch.load(encoder_path)
    print("Load encoder from {} done!".format(encoder_path))
    attn_decoder1 = torch.load(decoder_path)
    print('Load decoder from {} done!'.format(decoder_path))
    #

    # get generator
    x_paths = glob.glob(os.path.join(test_dir, '*.pt'))
    x_paths = sort_test_xpaths(x_paths)
    test_generator = torch_test_gen(x_paths)
    #

    # load vocab
    vocab = pickle.load(open(vocab_path, 'rb'))
    #

    # predict
    for i, src_tensor in enumerate(test_generator):
        decoded_words = predict(encoder1, attn_decoder1, src_tensor, vocab,
                                                                max_length=Y_max_length, SOS_token=0)
        save_path = os.path.join(save_dir, '{}.txt'.format(i + 1))

        # TODO, add language model
        print("id: {}, Decoded_words: {}".format(i, decoded_words))

        # replace SOS, EOS, and check no UNK
        assert '<UNK>' not in decoded_words
        decoded_words = [word for word in decoded_words if word != '<SOS>']
        decoded_words = [word for word in decoded_words if word != '<EOS>']
        #

        # save
        with open (save_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(decoded_words))