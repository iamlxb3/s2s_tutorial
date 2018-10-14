import pandas as pd
import random
import torch
import re
import ipdb
import numpy as np


class EnFraDataSet():
    def __init__(self, X_paths, Y_csv_path, input_shape, output_shape, ignore_index):
        self.Y_train = pd.read_csv(Y_csv_path).set_index('id').to_dict()['index'] # dictionary
        self.X_train = X_paths
        self.tensor_input_empty = torch.zeros(input_shape)
        self.tensor_output_empty = torch.ones(output_shape).type(torch.LongTensor) * ignore_index # TODO, change value


    def __getitem__(self, index):
        x_path = self.X_train[index]

        # get tensor_input
        tensor_input = self.tensor_input_empty.clone() # TODO, better way than clone?
        tensor_input_temp = torch.load(x_path).float()
        tensor_input_temp = torch.from_numpy(tensor_input_temp.detach().numpy())
        # padding

        # #####################################################
        # another way of padding
        # seq = torch.Tensor([1, 2, 3])  # seq of variable length
        # print(F.pad(seq, pad=(0, 2), mode='constant', value=0))
        # #####################################################

        tensor_input[:tensor_input_temp.shape[0]] = tensor_input_temp
        #

        # get tensor_output
        uid = int(re.findall(r'x_([0-9]+).pt', x_path)[0])
        y_indexes = self.Y_train[uid].split(',')
        tensor_output_temp = np.zeros((len(y_indexes), 1))
        for i, y_index in enumerate(y_indexes):
            tensor_output_temp[i] = int(y_index)
        tensor_output_temp = torch.from_numpy(tensor_output_temp)
        tensor_output_temp = tensor_output_temp.type(torch.LongTensor)
        tensor_output = self.tensor_output_empty.clone()
        tensor_output[:tensor_output_temp.shape[0]] = tensor_output_temp
        #

        return tensor_input, tensor_output, uid

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