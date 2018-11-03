import torch
import random
import os
import pickle
import sys
import pandas as pd
import argparse

sys.path.append("..")
from funcs.trainer import epoches_train
from utils.helpers import model_get

from funcs.gen import EnFraDataSet
from torch.utils.data import DataLoader
from funcs.recorder import EpochRecorder
import torch.nn as nn
from torch import optim
from config import config

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


    # read VOCAB
    en_vocab = pickle.load(open(en_vocab_path, 'rb'))
    fra_vocab = pickle.load(open(fra_vocab_path, 'rb'))

    src_vocab_len = len(en_vocab)
    # src_EOS_token = int(en_vocab.index('<EOS>'))
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
    N = 100
    if N is None:
        N = 999999999
    #

    # load model
    encoder, decoder = model_get(device, load_model, encoder_path, decoder_path, config, is_train=True)

    config.encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.lr, eps=1e-3, amsgrad=True)
    config.decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.lr, eps=1e-3, amsgrad=True)
    config.criterion = nn.NLLLoss(ignore_index=target_pad_token)
    #

    # Split train / val, TODO
    X = pd.read_csv(seq_csv_path)['source'].values[:N]
    Y = pd.read_csv(seq_csv_path)['target'].values[:N]
    uids = pd.read_csv(seq_csv_path)['uid'].values[:N]

    random.seed(1)  # TODO, add shuffle
    shuffled_X_Y_uids = list(zip(X, Y, uids))
    random.shuffle(shuffled_X_Y_uids)
    X, Y, uids = zip(*shuffled_X_Y_uids)
    val_percent = 0.2
    val_index = int(val_percent * len(X))
    train_X, train_Y, train_uids = X[val_index:], Y[val_index:], uids[val_index:]
    val_X, val_Y, val_uids = X[:val_index], Y[:val_index], uids[:val_index]
    #

    # get generator
    train_generator = EnFraDataSet(train_X, train_Y, train_uids, config.encoder_pad_shape, config.decoder_pad_shape,
                                   src_pad_token, target_pad_token, use_pretrain_embedding)
    train_loader = DataLoader(train_generator,
                              batch_size=config.batch_size,
                              shuffle=config.data_shuffle,
                              num_workers=config.num_workers,
                              # pin_memory=True
                              )
    val_generator = EnFraDataSet(val_X, val_Y, val_uids, config.encoder_pad_shape, config.decoder_pad_shape,
                                 src_pad_token, target_pad_token, use_pretrain_embedding)
    val_loader = DataLoader(val_generator,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers,
                            # pin_memory=True
                            )

    # get epoch recorder
    epoch_recorder = EpochRecorder()
    #

    # start training
    step_size = len(train_generator) / config.batch_size
    config.step_size = step_size
    epoches_train(config, train_loader, val_loader, encoder, decoder, epoch_recorder, encoder_path, decoder_path)
    print('Training done!')
    #

if __name__ == '__main__':
    main()
