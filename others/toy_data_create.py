import pandas as pd
import numpy as np
import random
import pickle
import ipdb
import os


def vocab_create():
    alphabat = 'abcdefghijklmnopqrstuvwxyz'
    vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + list(alphabat)
    pickle.dump(vocab, open('../data/s2s_toy_data_copy/vocab.pkl', 'wb'))
    pickle.dump(vocab, open('../data/s2s_toy_data_reverse/vocab.pkl', 'wb'))
    return vocab


def letters_to_index(vocab, letters):
    return [vocab.index(l) for l in letters]


def toy_data_create(N, alphabat, min_len, max_len, vocab, save_dir, reverse=False):
    df = {'source': [], 'target': [], 'uid': []}
    train_save_path = os.path.join(save_dir, 'train.csv')
    test_save_path = os.path.join(save_dir, 'test.csv')

    for i in range(N):
        sample = random.choices(alphabat, k=random.randint(min_len, max_len))
        src = letters_to_index(vocab, sample)
        if reverse:
            target = src[::-1]
        else:
            target = src.copy()
        src.append(vocab.index('<EOS>'))
        target.append(vocab.index('<EOS>'))
        src = ','.join([str(x) for x in src])
        target = ','.join([str(x) for x in target])
        df['source'].append(src)
        df['target'].append(target)
        df['uid'].append(i)

    df = pd.DataFrame(df)
    msk = np.random.rand(len(df)) < 0.9
    train = df[msk]
    test = df[~msk]

    train.to_csv(train_save_path, index=False)
    test.to_csv(test_save_path, index=False)
    print("Save train to {}, save test to {}".format(train_save_path, test_save_path))


def main():
    alphabat = list('abcdefghijklmnopqrstuvwxyz')
    min_len = 3
    max_len = 50
    N = 20000

    df = {'source': [], 'target': [], 'uid': []}
    vocab = vocab_create()

    # create copy toy data
    save_dir = '../data/s2s_toy_data_copy'
    toy_data_create(N, alphabat, min_len, max_len, vocab, save_dir, reverse=False)

    # create reverse toy data
    save_dir = '../data/s2s_toy_data_reverse'
    toy_data_create(N, alphabat, min_len, max_len, vocab, save_dir, reverse=True)


if __name__ == '__main__':
    main()
