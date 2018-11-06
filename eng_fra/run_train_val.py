"""
TODOLIST:
1. add attention display
2. clean code on eval/test

try pin_memory to speed up
"""
import random
import sys
import pandas as pd
import argparse

sys.path.append("..")
from funcs.trainer import epoches_train
from utils.helpers import model_get

from funcs.gen import EnFraDataSet
from torch.utils.data import DataLoader
from funcs.recorder import EpochRecorder
from torch.optim import lr_scheduler
from torch import optim
from config import cfg


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', action="store_true", help='Epoch size', default=False)
    args = parser.parse_args()
    return args


def main():
    # TODO, add parsing
    args = args_parse()
    #

    # TODO, temp, other configs
    N = 500
    if N is None:
        N = 999999999
    #

    # load model
    encoder, decoder = model_get(cfg)

    # config.encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.lr, eps=1e-3, amsgrad=True)
    # config.decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.lr, eps=1e-3, amsgrad=True)

    # set optimizer & lr_scheduler
    cfg.optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    cfg.lr_scheduler = lr_scheduler.ReduceLROnPlateau(cfg.optimizer, 'min', verbose=True, patience=3, min_lr=1e-8)
    #

    # Split train / val, TODO
    X = pd.read_csv(cfg.seq_csv_path)['source'].values[:N]
    Y = pd.read_csv(cfg.seq_csv_path)['target'].values[:N]
    uids = pd.read_csv(cfg.seq_csv_path)['uid'].values[:N]

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
    train_generator = EnFraDataSet(train_X, train_Y, train_uids, cfg.encoder_pad_shape, cfg.decoder_pad_shape,
                                   cfg.src_pad_token, cfg.target_pad_token, cfg.use_pretrain_embedding)
    train_loader = DataLoader(train_generator,
                              batch_size=cfg.batch_size,
                              shuffle=cfg.data_shuffle,
                              num_workers=cfg.num_workers,
                              # pin_memory=True
                              )
    val_generator = EnFraDataSet(val_X, val_Y, val_uids, cfg.encoder_pad_shape, cfg.decoder_pad_shape,
                                 cfg.src_pad_token, cfg.target_pad_token, cfg.use_pretrain_embedding)
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
    step_size = len(train_generator) / cfg.batch_size
    cfg.step_size = step_size
    epoches_train(cfg, train_loader, val_loader, encoder, decoder, epoch_recorder, cfg.encoder_path, cfg.decoder_path)
    print('Training done!')
    #


if __name__ == '__main__':
    main()
