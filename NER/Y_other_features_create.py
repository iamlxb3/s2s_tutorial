import os
import torch
import ipdb
import pickle
import numpy as np
import pandas as pd

def create_vocab(save_dir):

    temp = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']
    ner_list = []
    pos_list = []
    syntax_list = []
    train_txt_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/raw2/train.txt'
    test_txt_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/raw2/test.txt'

    with open(train_txt_path, 'r') as f:
        for line in f:
            if line.strip() and '-DOCSTART-' not in line:
                line_list = line.strip().split(' ')[1:]
                pos, syn, y = line_list
                ner_list.append(y)
                pos_list.append(pos)
                syntax_list.append(syn)

    with open(test_txt_path, 'r') as f:
        for line in f:
            if line.strip() and '-DOCSTART-' not in line:
                line_list = line.strip().split(' ')[1:]
                pos, syn, y = line_list
                ner_list.append(y)
                pos_list.append(pos)
                syntax_list.append(syn)

    ner_list = temp + sorted(list(set(ner_list)))
    pos_list = temp + sorted(list(set(pos_list)))
    syntax_list = temp + sorted(list(set(syntax_list)))


    pickle.dump(ner_list, open(os.path.join(save_dir, 'NER_vocab.pkl'), 'wb'))
    pickle.dump(pos_list, open(os.path.join(save_dir, 'POS_vocab.pkl'), 'wb'))
    pickle.dump(syntax_list, open(os.path.join(save_dir, 'syntax_vocab.pkl'), 'wb'))


def y_df_create(txt_path, ner_vocab, save_path):
    y_df = {'uid': [], 'target': []}
    index = 0
    with open(txt_path, 'r') as f:
        Y = []
        for line in f:
            if line.strip() and '-DOCSTART-' not in line:
                line_list = line.strip().split(' ')[1:]
                pos, syn, y = line_list
                Y.append(y)
            else:
                if Y:
                    y_df['uid'].append(index)
                    y_df['target'].append(','.join([str(ner_vocab.index(y)) for y in Y]))
                    index += 1
                    Y = []
    y_df = pd.DataFrame(y_df)
    y_df = y_df[['uid', 'target']]
    y_df.to_csv(save_path, index=False)

def other_features_create(txt_path, save_dir, pos_vocab, syn_vocab):

    index = 0
    with open(txt_path, 'r') as f:
        pos_list = []
        syntax_list = []
        for line in f:
            if line.strip() and '-DOCSTART-' not in line:
                line_list = line.strip().split(' ')[1:]
                pos, syn, y = line_list
                pos_list.append(pos)
                syntax_list.append(syn)
            else:
                if pos_list and syntax_list:
                    pos_tensor = np.zeros((len(pos_list), len(pos_vocab)))
                    syn_tensor = np.zeros((len(syntax_list), len(syn_vocab)))

                    for t, pos in enumerate(pos_list):
                        pos_tensor[t][pos_vocab.index(pos)] = 1.0

                    for t, syn in enumerate(syntax_list):
                        syn_tensor[t][syn_vocab.index(syn)] = 1.0

                    tensor = np.concatenate([pos_tensor, syn_tensor], axis=1)
                    x_save_path = os.path.join(save_dir, 'x_{}.npy'.format(index))
                    np.save(x_save_path, tensor)
                    index += 1

                pos_list = []
                syntax_list = []




if __name__ == '__main__':
    train_save_dir = '/Users/pjs/byte_play_main/data/CoNLL_2003/train'
    test_save_dir = '/Users/pjs/byte_play_main/data/CoNLL_2003/test'
    train_txt_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/raw2/train.txt'
    test_txt_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/raw2/test.txt'

    #main(glove_dict, train_save_dir, train_txt_path)

    save_dir = '/Users/pjs/byte_play_main/data/CoNLL_2003'

    # # (1.) create vocab
    # create_vocab(save_dir)

    # # (2.) create y
    # train_txt_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/raw2/train.txt'
    # test_txt_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/raw2/test.txt'
    # ner_vocab = pickle.load(open('/Users/pjs/byte_play_main/data/CoNLL_2003/NER_vocab.pkl', 'rb'))
    # train_save_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/train.csv'
    # test_save_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/test.csv'
    # y_df_create(train_txt_path, ner_vocab, train_save_path)
    # y_df_create(test_txt_path, ner_vocab, test_save_path)

    # (3.) create other features
    train_txt_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/raw2/train.txt'
    test_txt_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/raw2/test.txt'
    train_save_dir = '/Users/pjs/byte_play_main/data/CoNLL_2003/train'
    test_save_dir = '/Users/pjs/byte_play_main/data/CoNLL_2003/test'
    pos_vocab = pickle.load(open('/Users/pjs/byte_play_main/data/CoNLL_2003/POS_vocab.pkl', 'rb'))
    syn_vocab = pickle.load(open('/Users/pjs/byte_play_main/data/CoNLL_2003/syntax_vocab.pkl', 'rb'))
    other_features_create(train_txt_path, train_save_dir, pos_vocab, syn_vocab)
    other_features_create(test_txt_path, test_save_dir, pos_vocab, syn_vocab)