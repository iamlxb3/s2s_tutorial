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

from main_funcs.trainer import epoches_train
from utils.helpers import model_get
from main_funcs.gen import EnFraDataSet
from torch.utils.data import DataLoader

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
    save_model = True
    hidden_size = 256
    encoder_nlayers = 1
    input_dim = 300
    #

    # set path
    data_set = 'eng_fra'
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data')
    model_pkl_dir = os.path.join(top_dir, 'model_pkls')
    train_x_dir = os.path.join(data_dir, data_set, 'train')
    y_csv_path = os.path.join(data_dir, data_set, 'train_y.csv')
    vocab_path = os.path.join(data_dir, data_set, 'vocab.pkl')
    encoder_path = os.path.join(model_pkl_dir, '{}_encoder.pkl'.format(data_set))
    decoder_path = os.path.join(model_pkl_dir, '{}_decoder.pkl'.format(data_set))
    #

    # read VOCAB
    vocab = pickle.load(open(vocab_path, 'rb'))
    Vocab_len = len(vocab)
    EOS_index = int(vocab.index('<EOS>'))
    SOS_index = int(vocab.index('<SOS>'))
    ignore_index = Vocab_len
    print("Vocab size: {}, ignore_index: {}".format(Vocab_len, ignore_index))
    print("EOS_index: {}, SOS_index: {}".format(EOS_index, SOS_index))
    #

    # training config
    use_teacher_forcing = True
    N = 10000
    epoches = 28
    batch_size = 128
    max_length = 82
    num_workers = 8
    lr = 1e-3
    input_shape = (max_length, input_dim)  # this is for padding, batch training
    output_shape = (max_length, 1)  # this is for padding, batch training
    #

    # load model
    encoder, decoder = model_get(device, load_model, encoder_path, decoder_path, input_dim, hidden_size,
                                 encoder_nlayers, Vocab_len,
                                 max_length, is_train=True)
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
    train_generator = EnFraDataSet(train_x_paths, y_csv_path, input_shape, output_shape, ignore_index)
    train_loader = DataLoader(train_generator,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              # pin_memory=True
                              )

    # val_generator = EnFraDataSet(val_x_paths, y_csv_path, input_shape, output_shape, ignore_index)
    # val_loader = DataLoader(val_generator,
    #                         batch_size=batch_size,
    #                         shuffle=False,
    #                         num_workers=num_workers,
    #                         # pin_memory=True
    #                         )
    #

    # start training
    step_size = len(train_generator) / batch_size
    epoches_train(train_loader, encoder, decoder, epoches, step_size, EOS_index, SOS_index, ignore_index,
                  learning_rate=lr, verbose=True, use_teacher_forcing=use_teacher_forcing, device=device)
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
