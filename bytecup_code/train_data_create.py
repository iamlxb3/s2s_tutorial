import os
import pickle
import glob
import torch
import ipdb
import pandas as pd
import numpy as np
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import sys

sys.path.append('..')

from main_funcs.helpers import pos_encode




def main(elmo, batch_to_ids):

    vocab_size = 30000
    vocab_path = '/Users/pjs/byte_play/data/bytecup2018/bytecup_vocab.pkl'
    vocab = pickle.load(open(vocab_path, 'rb'))[:vocab_size]
    data_dir = '/Users/pjs/byte_play/data/bytecup2018'
    train_dir = '/Users/pjs/byte_play/data/bytecup2018/train'
    txt_paths = glob.glob(os.path.join(data_dir, 'bytecup.corpus.train.8.txt'))
    y_pd_path = '/Users/pjs/byte_play/data/bytecup2018/train_y.csv'

    df = {'id': [], 'w_index': [], 'title': []}
    counter = 0

    stop_words = set(stopwords.words('english'))


    for txt_path in txt_paths:
        with open(txt_path, 'r') as f:
            for i, line in enumerate(f):

                #step1_t1 = time.time()
                dict1 = eval(line)
                content = dict1['content']
                title = dict1['title']
                words1 = word_tokenize(content)
                content_words = [x.lower() for x in words1]
                content_words = [x for x in content_words if x not in stop_words]
                words2 = word_tokenize(title)
                title_words = [x.lower() for x in words2]
                title_words = ['<SOS>'] + title_words + ['<EOS>']
                # step1_t2 = time.time()
                # print("step1: ", step1_t2-step1_t1)

                #step2_t1 = time.time()
                # get the source tensor
                eng_ids = batch_to_ids([content_words])
                embeddings = elmo(eng_ids)
                source_tensor = embeddings['elmo_representations'][0][0]
                # step2_t2 = time.time()
                # print("step2: ", step2_t2 - step2_t1)


                # postion encoding
                #step3_t1 = time.time()
                pos_tensor = []
                for w_index, word in enumerate(content_words):
                    pos_tensor_t = pos_encode(w_index, 1024)
                    pos_tensor.append(pos_tensor_t)
                pos_tensor = np.concatenate(pos_tensor)
                pos_tensor = torch.from_numpy(pos_tensor)
                # step3_t2 = time.time()
                # print("step3: ", step3_t2 - step3_t1)
                #

                #step4_t1 = time.time()
                # add pos and elmo
                source_tensor = pos_tensor.float() + source_tensor
                #
                x_save_path = os.path.join(train_dir, 'x_{}.pt'.format(counter))
                torch.save(source_tensor, x_save_path)
                print("Save to {}".format(x_save_path))
                #

                # get the target tensor
                word_indexes = []
                for word in title_words:
                    try:
                        word_index = vocab.index(word)
                    except ValueError:
                        word_index = vocab.index('<UNK>')

                    word_indexes.append(str(word_index))
                #

                df['id'].append(counter)
                df['w_index'].append(','.join(word_indexes))
                df['title'].append(title_words)
                counter += 1

                # step4_t2 = time.time()
                # print("step4: ", step4_t2 - step4_t1)

                if i > 1000:
                    break
                print("-----------------------------")

    df = pd.DataFrame(df)
    df.to_csv(y_pd_path, index=False)
    print("Save to {}".format(y_pd_path))


def init_emlo():
    from allennlp.modules.elmo import Elmo, batch_to_ids

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway" \
                   "/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway" \
                  "/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    return elmo, batch_to_ids


if __name__ == '__main__':
    elmo, batch_to_ids = init_emlo()
    main(elmo, batch_to_ids)
