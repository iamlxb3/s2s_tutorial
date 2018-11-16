"""
训练函数，类似于main的功能
"""
import random
import sys
import pandas as pd
import argparse

sys.path.append("..")
from funcs.trainer import epoches_train
from utils.helpers import model_get

from funcs.gen import Seq2SeqDataSet
from torch.utils.data import DataLoader
from funcs.recorder import EpochRecorder
from torch.optim import lr_scheduler
from torch import optim
from config import cfg


def main():
    #
    N = 20  # TODO 这里是测试用的，把train的大小限制在20，如果你真的要跑，把N改成None
    if N is None:
        N = 999999999
    #

    # load model
    encoder, decoder = model_get(cfg)

    # set optimizer & lr_scheduler，设置优化器和scheduler，基本不动也没事儿~
    cfg.optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    cfg.lr_scheduler = lr_scheduler.ReduceLROnPlateau(cfg.optimizer, 'min', verbose=True, patience=3, min_lr=1e-6)
    #

    # Split train / val
    csv_path = cfg.train_seq_csv_path
    X = pd.read_csv(csv_path)['source'].values[:N]
    Y = pd.read_csv(csv_path)['target'].values[:N]
    uids = pd.read_csv(csv_path)['uid'].values[:N]

    random.seed(1)  # TODO, 是否用random seed，这里只保证val和train每次都被分割的一样
    shuffled_X_Y_uids = list(zip(X, Y, uids))
    random.shuffle(shuffled_X_Y_uids)
    X, Y, uids = zip(*shuffled_X_Y_uids)
    val_percent = 0.2
    val_index = int(val_percent * len(X))
    train_X, train_Y, train_uids = X[val_index:], Y[val_index:], uids[val_index:]
    val_X, val_Y, val_uids = X[:val_index], Y[:val_index], uids[:val_index]
    print("Split train/val done!")
    #

    # get generator 生成train/val的data loader，类似于生成器的功能
    train_generator = Seq2SeqDataSet(train_X, train_Y, train_uids, cfg.encoder_pad_shape, cfg.decoder_pad_shape,
                                     cfg.src_pad_token, cfg.target_pad_token, cfg.use_pretrain_embedding)
    train_loader = DataLoader(train_generator,
                              batch_size=cfg.batch_size,
                              shuffle=cfg.data_shuffle,
                              num_workers=cfg.num_workers,
                              # pin_memory=True
                              )
    val_generator = Seq2SeqDataSet(val_X, val_Y, val_uids, cfg.encoder_pad_shape, cfg.decoder_pad_shape,
                                   cfg.src_pad_token, cfg.target_pad_token, cfg.use_pretrain_embedding)
    val_loader = DataLoader(val_generator,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            # pin_memory=True
                            )

    # get epoch recorder，初始化EpochRecorder
    epoch_recorder = EpochRecorder()
    #

    # start training，开始训练
    step_size = len(train_generator) / cfg.batch_size
    cfg.step_size = step_size
    epoches_train(cfg, train_loader, val_loader, encoder, decoder, epoch_recorder, cfg.encoder_path, cfg.decoder_path)
    print('Training done!')
    #


if __name__ == '__main__':
    main()
