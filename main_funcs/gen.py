import pandas as pd
import random
import torch
import re
import ipdb
import numpy as np


class EnFraDataSet():
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


def torch_random_train_gen(X_paths, Y_csv_path, index_name='w_index'):
    # get y_dict
    y_df = pd.read_csv(Y_csv_path)
    y_dict = y_df.set_index('id').to_dict()[index_name]
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


def torch_test_gen(X_paths):
    for x_path in X_paths:
        tensor_input = torch.load(x_path)
        yield tensor_input


def torch_val_gen(X_paths, Y_csv_path, index_name='w_index'):
    # get y_dict
    y_df = pd.read_csv(Y_csv_path)
    y_dict = y_df.set_index('id').to_dict()[index_name]
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

        yield (tensor_input, tensor_output, id)

# def train_gen():
#     training_pairs = []
#     N = 50
#     for i in range(N):
#         arr_input = np.random.random((5, 256))
#         arr_output = np.zeros((5, 2925))
#         for t_arr in arr_output:
#             t_arr[random.randint(0, 2925)] = 1
#
#         # np.array input -> torch
#         tensor_input = torch.from_numpy(arr_input)
#         tensor_input = tensor_input.float()
#
#         # np.array output -> torch
#         tensor_output = torch.from_numpy(arr_output)
#         tensor_output = tensor_output.type(torch.LongTensor)
#
#         training_pairs.append((tensor_input, tensor_output))
#
#     while True:
#         for training_pair in training_pairs:
#             yield training_pair
