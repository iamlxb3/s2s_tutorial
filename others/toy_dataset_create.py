import os
import random
import torch
import pickle
import shutil
import pandas as pd
import ipdb


def indices_to_one_hot(indices, dim):
    tensor = torch.zeros((len(indices), dim), dtype=torch.float64)
    for i, index in enumerate(indices):
        tensor[i][index] = 1.0
    return tensor


def toy_data3(save_dir, is_clear):
    input_dim = 100
    N = 10000
    test_N_index = 9000
    train_dir = os.path.join(save_dir, 'train')
    test_dir = os.path.join(save_dir, 'test')
    if is_clear:
        shutil.rmtree(train_dir)
        shutil.rmtree(test_dir)
        os.makedirs(train_dir)
        os.makedirs(test_dir)
    Y_train = {'uid': [], 'target': [], 'source': []}
    Y_test = {'uid': [], 'target': [], 'source': []}

    for uid in range(N):
        # get y
        is_reverse = random.choice([True, False])
        start_int = random.randint(0, input_dim - 20)
        end_int = min(input_dim - 1, start_int + random.randint(3, 50))
        indices = [x for x in range(start_int, end_int)]

        if is_reverse:
            indices = indices[::-1]
            y = 0
        else:
            y = 1

        tensor = indices_to_one_hot(indices, input_dim)

        if uid >= test_N_index:
            Y_test['uid'].append(uid)
            Y_test['source'].append(','.join([str(x) for x in indices]))
            Y_test['target'].append(y)
            torch.save(tensor, os.path.join(test_dir, '{}.pt'.format(uid)))
        else:
            Y_train['uid'].append(uid)
            Y_train['source'].append(','.join([str(x) for x in indices]))
            Y_train['target'].append(y)
            torch.save(tensor, os.path.join(train_dir, '{}.pt'.format(uid)))
        print(uid)

    Y_train = pd.DataFrame(Y_train)
    Y_train = Y_train[['uid', 'source', 'target']]
    Y_test = pd.DataFrame(Y_test)
    Y_test = Y_test[['uid', 'source', 'target']]
    Y_train.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
    Y_test.to_csv(os.path.join(save_dir, 'test.csv'), index=False)

    # save dummy vocab
    vocab = [0, 1] + ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    pickle.dump(vocab, open(os.path.join(save_dir, 'vocab.pkl'), 'wb'))
    #

def toy_data4(save_dir, is_clear):
    """
    Copy the first and last 2 elements of the input
    """
    input_dim = 100
    N = 10000
    test_N_index = 9000
    train_dir = os.path.join(save_dir, 'train')
    test_dir = os.path.join(save_dir, 'test')
    if is_clear:
        if os.path.isdir(train_dir):
            shutil.rmtree(train_dir)
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(train_dir)
        os.makedirs(test_dir)
    Y_train = {'uid': [], 'target': [], 'source': []}
    Y_test = {'uid': [], 'target': [], 'source': []}

    for uid in range(N):
        # get y
        start_int = random.randint(0, input_dim - 20)
        end_int = min(input_dim - 1, start_int + random.randint(5, 50))
        indices = [x for x in range(start_int, end_int)]

        is_reverse = random.choice([True, False])
        if is_reverse:
            indices = indices[::-1]

        source_indices = ','.join([str(x) for x in indices])
        if is_reverse:
            target_indices = indices[-2:] + indices[:2]
        else:
            target_indices = indices[:2] + indices[-2:]
        target_indices = ','.join([str(x) for x in target_indices])

        assert len(indices) >= 4

        tensor = indices_to_one_hot(indices, input_dim)

        if uid >= test_N_index:
            Y_test['uid'].append(uid)
            Y_test['source'].append(source_indices)
            Y_test['target'].append(target_indices)
            torch.save(tensor, os.path.join(test_dir, '{}.pt'.format(uid)))
        else:
            Y_train['uid'].append(uid)
            Y_train['source'].append(source_indices)
            Y_train['target'].append(target_indices)
            torch.save(tensor, os.path.join(train_dir, '{}.pt'.format(uid)))
        print(uid)


    Y_train = pd.DataFrame(Y_train)
    Y_train = Y_train[['uid', 'source', 'target']]
    Y_test = pd.DataFrame(Y_test)
    Y_test = Y_test[['uid', 'source', 'target']]
    Y_train.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
    Y_test.to_csv(os.path.join(save_dir, 'test.csv'), index=False)

    # save dummy vocab
    vocab = list(range(0, input_dim)) + ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    pickle.dump(vocab, open(os.path.join(save_dir, 'vocab.pkl'), 'wb'))
    #

if __name__ == '__main__':

    data3_save_dir = '/Users/pjs/byte_play_main/data/s2s_toy_data_input_no_embedding'
    data4_save_dir = '/Users/pjs/byte_play_main/data/toy_data4'

    toy_data3(data3_save_dir, True)
    toy_data4(data4_save_dir, True)