import numpy as np
import torch
import random
import ipdb
import glob
import os
import pickle
import sys
import pandas as pd
import argparse
from easydict import EasyDict as edict

sys.path.append("..")
from main_funcs.trainer import epoches_train2
from utils.helpers import model_get
from utils.helpers import seq_max_length_get
from main_funcs.gen import EnFraDataSet
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', action="store_true", help='Epoch size', default=False)
    args = parser.parse_args()
    return args


def main():
    # TODO, add parsing
    args = args_parse()
    #

    # model config
    load_model = False
    use_pretrain_embedding = False
    save_model = True
    #

    # set path
    data_set = 'eng_fra'
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data')
    model_pkl_dir = os.path.join(top_dir, 'model_pkls')
    train_x_dir = os.path.join(data_dir, data_set, 'train')
    seq_csv_path = os.path.join(data_dir, data_set, 'train_small_seq.csv')
    en_vocab_path = os.path.join(data_dir, data_set, 'small_eng_vocab.pkl')
    fra_vocab_path = os.path.join(data_dir, data_set, 'small_fra_vocab.pkl')
    encoder_path = os.path.join(model_pkl_dir, '{}_encoder.pkl'.format(data_set))
    decoder_path = os.path.join(model_pkl_dir, '{}_decoder.pkl'.format(data_set))
    #

    # config dict
    config = edict()
    config.device = device
    config.encoder_input_dim = 256
    config.encoder_hidden_dim = 256
    config.decoder_input_dim = 256
    config.decoder_hidden_dim = 256
    config.encoder_pad_shape = (seq_max_length_get(seq_csv_path, 'source'), 1)
    config.decoder_pad_shape = (seq_max_length_get(seq_csv_path, 'target'), 1)
    config.lr = 1e-3
    config.epoches = 28
    config.batch_size = 32
    config.num_workers = 8
    config.use_teacher_forcing = True
    config.data_shuffle = True
    config.verbose = True
    #

    # read VOCAB
    en_vocab = pickle.load(open(en_vocab_path, 'rb'))
    fra_vocab = pickle.load(open(fra_vocab_path, 'rb'))

    src_vocab_len = len(en_vocab)
    #src_EOS_token = int(en_vocab.index('<EOS>'))
    src_pad_token = int(en_vocab.index('<PAD>'))

    target_vocab_len = len(fra_vocab)
    target_SOS_token = int(fra_vocab.index('<SOS>'))
    target_EOS_token = int(fra_vocab.index('<EOS>'))
    target_pad_token = int(fra_vocab.index('<PAD>'))

    config.src_vocab_len = src_vocab_len
    config.target_vocab_len = target_vocab_len
    config.src_pad_token = src_pad_token
    config.target_SOS_token = target_SOS_token
    config.target_EOS_token = target_EOS_token
    config.target_pad_token = target_pad_token
    #

    # other configs
    N = 10000
    #

    # load model
    encoder, decoder = model_get(device, load_model, encoder_path, decoder_path, config, is_train=True)

    config.encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.lr, eps=1e-3, amsgrad=True)
    config.decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.lr, eps=1e-3, amsgrad=True)
    config.criterion = nn.NLLLoss(ignore_index=target_pad_token)
    #



    # Split train / val
    x_paths = glob.glob(os.path.join(train_x_dir, '*.pt'))
    print("Total: ", len(x_paths))
    x_paths = x_paths[:N]
    random.seed(1)  # TODO, add shuffle
    random.shuffle(x_paths)
    val_percent = 0.2
    val_index = int(N * val_percent)
    train_x_paths = x_paths[val_index:N]
    val_x_paths = x_paths[0:val_index]
    #

    # get generator
    train_generator = EnFraDataSet(seq_csv_path, config.encoder_pad_shape, config.decoder_pad_shape,
                                   src_pad_token, target_pad_token, use_pretrain_embedding)
    train_loader = DataLoader(train_generator,
                              batch_size=config.batch_size,
                              shuffle=config.data_shuffle,
                              num_workers=config.num_workers,
                              # pin_memory=True
                              )

    # val_generator = EnFraDataSet(val_x_paths, seq_csv_path, input_shape, output_shape, ignore_index)
    # val_loader = DataLoader(val_generator,
    #                         batch_size=batch_size,
    #                         shuffle=False,
    #                         num_workers=num_workers,
    #                         # pin_memory=True
    #                         )
    #

    # start training
    step_size = len(train_generator) / config.batch_size
    config.step_size = step_size
    epoches_train2(config, train_loader, encoder, decoder)
    print('training ok!')
    #

    if save_model:
        # save model
        torch.save(encoder, encoder_path)
        print("Save encoder to {}.".format(encoder_path))
        torch.save(decoder, decoder_path)
        print("Save decoder to {}.".format(decoder_path))
    #


if __name__ == '__main__':
    main()
