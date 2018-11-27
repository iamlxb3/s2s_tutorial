import pandas as pd
import random
import torch
import re
import ipdb
import os
import sys
import numpy as np


class Seq2SeqDataSet():
    def __init__(self, cfg, X, Y, uids, npy_dir=None):
        # TODO, optimize interface
        self.X_train = X
        self.Y_train = Y
        self.uids = uids
        self.is_index_input = cfg.is_index_input
        self.npy_dir = npy_dir

        if self.is_index_input:
            self.x_all_padding = torch.ones(cfg.x_pad_shape).type(torch.LongTensor) * cfg.src_pad_token
        else:
            # TODO, is zeros a good idea?
            self.x_all_padding = torch.zeros(cfg.x_pad_shape, dtype=torch.float32)
        self.y_all_padding = torch.ones(cfg.y_pad_shape).type(torch.LongTensor) * cfg.target_pad_token
        self.use_pretrain_embedding = cfg.use_pretrain_embedding

    def _pad_seq(self, seq_array, all_padding_seq):
        if len(seq_array.shape) == 1:
            seq_array = np.expand_dims(seq_array, axis=1)
        seq_temp = torch.from_numpy(seq_array).type(torch.LongTensor)
        seq = all_padding_seq.clone()
        seq[: seq_temp.shape[0]] = seq_temp
        return seq

    def __getitem__(self, index):
        """
        :return: x: torch.Size([max_padding_len, 1]), y: torch.Size([max_padding_len, 1])
        """
        # get tensor_output
        uid = self.uids[index]
        y_indices = np.array([int(y) for y in self.Y_train[index].split(',')])
        y = self._pad_seq(y_indices, self.y_all_padding)

        if self.is_index_input:
            x_indices = np.array([int(x) for x in self.X_train[index].split(',')])
            x = self._pad_seq(x_indices, self.x_all_padding)
        else:
            npy_path = os.path.join(self.npy_dir, "{}.npy".format(str(uid)))
            arr = np.load(npy_path)
            x = torch.from_numpy(arr)
            # TODO, remove assert
            assert x.size(0) <= self.x_all_padding.size(0)
            assert x.size(1) == self.x_all_padding.size(1)
            tmp_x = self.x_all_padding.clone()
            tmp_x[:x.shape[0]] = x
            x = tmp_x
        return x, y, uid

    def __len__(self):
        return len(self.X_train)
