import pandas as pd
import numpy as np
import pickle
import ntpath
import string
import torch
import glob
import ipdb
import sys
import re

sys.path.append('..')

from utils.bytecup_helpers import process_sentence, process_sentence_for_body, read_all_names
import os


def words_2_index(words, vocab, names, punctuations, unk_index, eos_index):
    indices = []
    for word in words:
        if not word.isalpha() and word not in punctuations:
            indices.append(vocab.index('<NOTALPHA>'))
        # elif word in names:
        #     indices.append(vocab.index('<NAME>'))
        else:
            try:
                index = vocab.index(word)
            except:
                index = unk_index
            indices.append(index)
    indices.append(eos_index)
    indices = ','.join(indices)
    return indices


def save_x_y_npy(vocab, glove_dict=None):

    # ['<NAME>', '<NOTALPHA>', '<UNK>', '<SOS>', '<EOS>', '', 'the', ',', 'to', 'of']

    punctuations = set(string.punctuation)
    names = read_all_names()

    txts = glob.glob(os.path.join('/Users/pjs/byte_play/data/bytecup2018/', '*.txt'))
    txts = [txt for txt in txts if re.findall(r'[0-9]+', ntpath.basename(txt))]

    train_csv_path = '/Users/pjs/byte_play/data/bytecup2018/train.csv'
    test_csv_path = '/Users/pjs/byte_play/data/bytecup2018/test.csv'

    V = len(vocab)
    unk_index = vocab.index('<UNK>')
    eos_index = vocab.index('<EOS>')

    df = {'uid': [], 'source': [], 'target': []}

    for txt in txts:
        with open(txt, 'r') as f:
            for i, line in enumerate(f):
                line_dict = eval(line)

                content = line_dict.get('content', '')
                title = line_dict.get('title', '')
                uid = line_dict['id']

                content_words = process_sentence_for_body(content)
                title_words = process_sentence(title)

                content_indices = words_2_index(content_words, vocab, names, punctuations, unk_index, eos_index)
                title_indices = words_2_index(title_words, vocab, names, punctuations, unk_index, eos_index)
                df['uid'].append(uid)
                df['source'].append(content_indices)
                df['target'].append(title_indices)

                if i % 100 == 0:
                    print("{} done".format(i))


    df = pd.DataFrame(df)
    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk]
    test_df = df[~msk]

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    # for test, TODO, add more in the future
    df = {'uid': [], 'source': []}
    val_set_txt_path = '/Users/pjs/byte_play/data/bytecup2018/bytecup.corpus.validation_set.txt'
    val_set_save_path = '/Users/pjs/byte_play/data/bytecup2018/validation_set.csv'

    with open(val_set_txt_path, 'r') as f:
        for i, line in enumerate(f):
            line_dict = eval(line)

            content = line_dict.get('content', '')
            uid = line_dict['id']
            content_words = process_sentence_for_body(content)
            content_indices = words_2_index(content_words, vocab, names, punctuations, unk_index, eos_index)
            df['uid'].append(uid)
            df['source'].append(content_indices)

    df = pd.DataFrame(df)
    df.to_csv(val_set_save_path, index=False)
    print("Save bytecup.corpus.validation_set csv to {}".format(val_set_save_path))
    #

def glove_dict_get():
    glove_txt_path = '/Users/pjs/byte_play/embeddings/glove.840B.300d.txt'
    glove_dict = {}
    with open(glove_txt_path, 'r') as f:
        for i, line in enumerate(f):
            line_split = line.split(' ')
            word = line_split[0]
            embedding = torch.tensor([float(x) for x in line_split[1:]])
            glove_dict[word] = embedding

            # if i == 100000:
            #     break
    print("read glove_dict complete!, Size: {}".format(len(glove_dict.keys())))
    return glove_dict  # 2196016


if __name__ == '__main__':
    vocab = pickle.load(open('/Users/pjs/byte_play/data/bytecup2018/vocab.pkl', 'rb'))
    save_x_y_npy(vocab)
