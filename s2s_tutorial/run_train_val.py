"""
TODOLIST:
1. see https://github.com/ymfa/seq2seq-summarizer    pointer-generator source code


try pin_memory to speed up
"""
import ipdb
import random
import math
import copy
import sys
import os
import torch
import pandas as pd
import argparse

sys.path.append("..")
from funcs.trainer import epoches_train
from utils.helpers import model_get
from utils.helpers import plot_results
from utils.helpers import output_config

from funcs.gen import Seq2SeqDataSet
from torch.utils.data import DataLoader
from funcs.recorder import EpochRecorder
from torch.optim import lr_scheduler
from torch import optim
from exp_config import experiments


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', help='the name of the experiment', type=str, required=False,
                        choices=list(experiments.keys()))
    args = parser.parse_args()
    return args


def main():
    # load experiment
    args = args_parse()
    if args.e is not None:
        cfg = experiments[args.e]
    else:
        from s2s_config import cfg
    #

    # load model
    encoder, decoder = model_get(cfg)

    # set optimizer & lr_scheduler
    cfg.optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr)
    cfg.lr_scheduler = lr_scheduler.ReduceLROnPlateau(cfg.optimizer, 'min', verbose=True, patience=3, min_lr=1e-6)
    #

    # Split train / val, TODO
    csv_path = cfg.train_seq_csv_path
    df = pd.read_csv(csv_path)
    tmp_df = df['source'].apply(lambda x: len(x.split(',')))
    mask = (tmp_df <= cfg.seq_max_len) & (tmp_df >= cfg.seq_min_len)  # TODO, filter by length
    df = df[mask]

    uids = df['uid'].values
    Y = df['target'].values
    X = df['source'].values if cfg.is_index_input else copy.copy(uids)

    random.seed(cfg.randseed)  # TODO, add shuffle
    torch.manual_seed(cfg.randseed)
    shuffled_X_Y_uids = list(zip(X, Y, uids))
    random.shuffle(shuffled_X_Y_uids)
    X, Y, uids = zip(*shuffled_X_Y_uids)
    val_percent = cfg.val_percent
    val_index = int(val_percent * len(X))
    train_X, train_Y, train_uids = X[val_index:], Y[val_index:], uids[val_index:]
    val_X, val_Y, val_uids = X[:val_index], Y[:val_index], uids[:val_index]
    #

    # get generator
    train_generator = Seq2SeqDataSet(cfg, train_X, train_Y, train_uids)
    train_loader = DataLoader(train_generator,
                              batch_size=cfg.batch_size,
                              shuffle=cfg.data_shuffle,
                              num_workers=cfg.num_workers,
                              # pin_memory=True
                              )
    val_generator = Seq2SeqDataSet(cfg, val_X, val_Y, val_uids)
    val_loader = DataLoader(val_generator,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            # pin_memory=True
                            )

    # get epoch recorder
    epoch_recorder = EpochRecorder()
    #

    # start training
    step_size = int(math.ceil(len(train_generator) / cfg.batch_size))
    cfg.step_size = step_size
    epoches_train(cfg, train_loader, val_loader, encoder, decoder, epoch_recorder, cfg.encoder_path, cfg.decoder_path)
    print('Training done!')
    #

    # plot results
    if cfg.plot_loss:
        if not os.path.isdir(cfg.exp_dir):
            os.makedirs(cfg.exp_dir)
        plot_results(epoch_recorder, title='', save_path=os.path.join(cfg.exp_dir, 'loss.png'))
    #

    # save config results
    output_config(cfg, os.path.join(cfg.exp_dir, 'config.csv'))
    #


if __name__ == '__main__':
    main()
