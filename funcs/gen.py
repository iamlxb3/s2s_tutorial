"""
数据的生成器，会设定输入和输出的最大长度，因为pytorch的data loader会做concatanate，又会打乱，所以必须要定长。
看下就行，不用改
"""

import pandas as pd
import random
import torch
import re
import ipdb
import numpy as np


class Seq2SeqDataSet():
    def __init__(self, X, Y, uids, x_pad_shape, y_pad_shape, src_pad_token, target_pad_token,
                 use_pretrain_embedding):
        self.X_train = X
        self.Y_train = Y
        self.uids = uids
        self.x_all_padding = torch.ones(x_pad_shape).type(torch.LongTensor) * src_pad_token
        self.y_all_padding = torch.ones(y_pad_shape).type(torch.LongTensor) * target_pad_token
        self.use_pretrain_embedding = use_pretrain_embedding

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
        x_indices = np.array([int(x) for x in self.X_train[index].split(',')])
        y_indices = np.array([int(y) for y in self.Y_train[index].split(',')])

        x = self._pad_seq(x_indices, self.x_all_padding)
        y = self._pad_seq(y_indices, self.y_all_padding)

        return x, y, uid

    def __len__(self):
        return len(self.X_train)