import os
import pandas as pd
import numpy as np
import pickle
import ipdb


def save_arr(df, vocab, save_dir):
    for i, row in df.iterrows():
        uid = row['uid']
        source = row['source']
        arr = np.zeros((len(source), len(vocab)))
        for i, value in enumerate(source):
            arr[i][int(value)] = 1.0
        np.save(os.path.join(save_dir, str(uid)), arr)


def main():
    data_dir = '/Users/pjs/byte_play_main/data/s2s_toy_data_reverse'

    train_save_dir = os.path.join(data_dir, 'train')
    test_save_dir = os.path.join(data_dir, 'test')
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    vocab_path = os.path.join(data_dir, 'vocab.pkl')
    vocab = pickle.load(open(vocab_path, 'rb'))

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df['source'] = train_df['source'].apply(lambda x: x.split(','))
    test_df['source'] = test_df['source'].apply(lambda x: x.split(','))

    save_arr(train_df, vocab, train_save_dir)
    save_arr(test_df, vocab, test_save_dir)


if __name__ == '__main__':
    main()
