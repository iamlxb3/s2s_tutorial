import ipdb
import re
import glob
import numpy as np
import pickle
import collections
import os
import sys

sys.path.append('..')
import pandas as pd
import torch
from utils.bytecup_helpers import process_sentence


def get_vocab():
    vocab_save_path = '/Users/pjs/byte_play/data/bytecup2018/vocab.pkl'
    vocab_set = []
    txts = glob.glob(os.path.join('/Users/pjs/byte_play/data/bytecup2018/', '*.txt'))

    for txt in txts:
        with open(txt, 'r') as f:
            for index, line in enumerate(f):
                line_dict = eval(line)
                content = line_dict.get('content', '')
                title = line_dict.get('title', '')

                words = content + title
                words = process_sentence(words)
                vocab_set.extend(words)

                if index % 1000 == 0:
                    print('vocab_set: ', len(vocab_set))
        print("{} done!".format(txt))

    vocab_list = collections.Counter(vocab_set).items()
    vocab_list = sorted(vocab_list, key=lambda x: x[1], reverse=True)[0:80000]
    vocab_list = ['<UNK>', '<SOS>', '<EOS>'] + [x[0] for x in vocab_list]
    pickle.dump(vocab_list, open(vocab_save_path, 'wb'))





if __name__ == '__main__':

    # elmo, batch_to_ids = init_emlo()
    get_vocab()

    # glove_dict = glove_dict_get()
    # vocab = pickle.load(open('eng_fra/vocab.pkl', 'rb'))
    # save_x_y_npy(vocab, glove_dict)
