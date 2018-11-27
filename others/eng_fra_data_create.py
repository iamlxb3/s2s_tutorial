import ipdb
from nltk.tokenize import word_tokenize
import re
import numpy as np
import pickle
import collections
import os
import pandas as pd
import torch


# # use batch_to_ids to convert sentences to character ids
# sentences = [['First', 'sentence', 'is', 'me', '.'], ['Ha']]
# character_ids = batch_to_ids(sentences)
# # embeddings['elmo_representations'][1].detach().numpy().shape -> numpy
# embeddings = elmo(character_ids)


def get_vocab():
    vocab_save_path = 'eng_fra/vocab.pkl'
    vocab_set = []

    with open('eng_fra/eng-fra.txt', 'r') as f:
        for index, line in enumerate(f):

            split_index_found = False
            for i, char in enumerate(line):
                if not split_index_found and (char == '.' or char == '!' or char == '?'):
                    split_index = i
                    split_index_found = True
            fra_sen = line[split_index + 1:].strip()
            fra_words = word_tokenize(fra_sen, language='french')
            fra_words = [x.lower() for x in fra_words]
            vocab_set.extend(fra_words)

            if index % 1000 == 0:
                print('vocab_set: ', len(vocab_set))

    vocab_list = collections.Counter(vocab_set).items()
    vocab_list = sorted(vocab_list, key=lambda x: x[1], reverse=True)[0:5000]
    vocab_list = ['<UNK>', '<SOS>', '<EOS>'] + [x[0] for x in vocab_list]

    ipdb.set_trace()
    pickle.dump(vocab_list, open(vocab_save_path, 'wb'))


def save_x_y_npy(vocab, glove_dict):
    save_dir = 'eng_fra/train'
    y_pd_path = 'eng_fra/train_y.csv'
    V = len(vocab)
    df = {'id': [], 'index': []}


    with open('eng_fra/eng-fra.txt', 'r') as f:
        for index, line in enumerate(f):

            # x_save_path = os.path.join(save_dir, 'x_{}.pt'.format(index))
            #
            split_index_found = False
            for i, char in enumerate(line):
                if not split_index_found and (char == '\t'):
                    split_index = i
                    split_index_found = True

            eng_sen = line[:split_index].strip()
            fra_sen = line[split_index:].strip()

            # eng_words = word_tokenize(eng_sen, language='english')
            fra_words = word_tokenize(fra_sen, language='french')

            fra_words = [x.lower() for x in fra_words]


            # for i, word in enumerate(fra_words):
            #     if '-' in word:
            #         print("Before: ", fra_words)
            #         fra_words.pop(i)
            #         fra_words[i:i] = word.split('-')
            #         print("After: ", fra_words)

            #fra_words = ['<SOS>'] + fra_words + ['<EOS>']
            fra_words = fra_words + ['<EOS>']


            # get the target tensor
            word_indexes = []
            for word in fra_words:
                try:
                    word_index = vocab.index(word)
                except:
                    word_index = vocab.index('<UNK>')
                word_indexes.append(str(word_index))


            df['id'].append(index)
            df['index'].append(','.join(word_indexes))
            #

            if index % 100 == 0:
                print("Index: ", index)

            # get the source tensor
            src_tensor = []
            content_words = word_tokenize(eng_sen)
            content_words = [x.lower() for x in content_words]

            for i, word in enumerate(content_words):
                if '-' in word:
                    print("Before: ", content_words)
                    content_words.pop(i)
                    content_words[i:i] = word.split('-')
                    print("After: ", content_words)

            for word in content_words:
                embedding = glove_dict.get(word, None)
                if embedding is None :
                    embedding = torch.zeros(300)
                    print("{} has no embedding! Sentence: {}".format(word, eng_sen))
                embedding = embedding.view(1, -1)
                src_tensor.append(embedding)

            src_tensor = torch.cat(src_tensor, 0)
            x_save_path = os.path.join(save_dir, 'x_{}.pt'.format(index))
            torch.save(src_tensor, x_save_path)
            #

            if index % 100 == 0:
                print("Index: {}".format(index))

            #
            #
            # print("\n")
            # print('English: ', eng_words)
            # print('French: ', fra_words)

    df = pd.DataFrame(df)
    df.to_csv(y_pd_path, index=False)


# def init_emlo():
#     from allennlp.modules.elmo import Elmo, batch_to_ids
#
#     options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway" \
#                    "/elmo_2x4096_512_2048cnn_2xhighway_options.json"
#     weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway" \
#                   "/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
#     elmo = Elmo(options_file, weight_file, 1, dropout=0)
#     return elmo, batch_to_ids

def glove_dict_get():
    glove_txt_path = '/Users/pjs/byte_play/embeddings/glove.840B.300d.txt'
    glove_dict = {}
    with open (glove_txt_path, 'r') as f:
        for i, line in enumerate(f):
            line_split = line.split(' ')
            word = line_split[0]
            embedding = torch.tensor([float(x) for x in line_split[1:]])
            glove_dict[word] = embedding

            # if i == 100000:
            #     break
    print("read glove_dict complete!, Size: {}".format(len(glove_dict.keys())))
    return glove_dict # 2196016

if __name__ == '__main__':
    # elmo, batch_to_ids = init_emlo()
    #get_vocab()

    glove_dict = glove_dict_get()
    vocab = pickle.load(open('eng_fra/vocab.pkl', 'rb'))
    save_x_y_npy(vocab, glove_dict)
