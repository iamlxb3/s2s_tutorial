
import ipdb
import collections
import pickle
import torch
import pandas as pd
import os
from nltk.tokenize import word_tokenize

def _sort_vocab(vocab):
    vocab = list(collections.Counter(vocab).items())
    vocab = sorted(vocab, key=lambda x: x[1], reverse=True)
    return vocab

def create_vocab(eng_vocab_path, fra_vocab_path):
    txt_path = '/Users/pjs/byte_play/data/eng_fra/eng-fra-small.txt'
    eng_vocab = []
    fra_vocab = []

    # create vocab first
    with open(txt_path, 'r') as f:
        for line in f:
            eng, fra = line.split('\t')
            eng_words = word_tokenize(eng, language='english')
            fra_words = word_tokenize(fra, language='french')
            eng_vocab.extend(eng_words)
            fra_vocab.extend(fra_words)

    eng_vocab = _sort_vocab(eng_vocab)
    fra_vocab = _sort_vocab(fra_vocab)

    eng_vocab = tuple(['<PAD>'] + ['<SOS>'] + ['<EOS>'] + ['<UNK>'] + [x[0] for x in eng_vocab])
    fra_vocab = tuple(['<PAD>'] + ['<SOS>'] + ['<EOS>'] + ['<UNK>'] + [x[0] for x in fra_vocab])

    pickle.dump(eng_vocab, open(eng_vocab_path, 'wb'))
    pickle.dump(fra_vocab, open(fra_vocab_path, 'wb'))

    print("Save vocab done!")

def glove_dict_get():
    glove_txt_path = '/Users/pjs/byte_play/embeddings/glove.840B.300d.txt'
    glove_dict = {}
    with open (glove_txt_path, 'r') as f:
        for i, line in enumerate(f):
            line_split = line.split(' ')
            word = line_split[0]
            embedding = torch.tensor([float(x) for x in line_split[1:]])
            glove_dict[word] = embedding

            if i == 950000:
                break
    print("read glove_dict complete!, Size: {}".format(len(glove_dict.keys())))
    return glove_dict # 2196016


def _words_to_indices(words, vocab):
    word_indices = []
    for word in words:
        try:
            word_index = vocab.index(word)
        except:
            word_index = vocab.index('<UNK>')
        word_indices.append(str(word_index))
    return word_indices

def save_x_y_npy(eng_vocab, fra_vocab, txt_path, glove_dict):

    save_dir = '/Users/pjs/byte_play/data/eng_fra/train_small'
    pd_path = '/Users/pjs/byte_play/data/eng_fra/train_small_seq.csv'
    df = {'uid': [], 'source': [], 'target': []}

    EOS_token = '<EOS>'
    EOS_vector = torch.zeros(300)
    torch.manual_seed(1)
    UNK_vector = torch.rand(300)

    with open(txt_path, 'r') as f:
        for index, line in enumerate(f):

            eng, fra = line.split('\t')
            eng_words = word_tokenize(eng, language='english')
            fra_words = word_tokenize(fra, language='french')

            eng_words.append(EOS_token)
            fra_words.append(EOS_token)

            eng_indices = _words_to_indices(eng_words, eng_vocab)
            fra_indices = _words_to_indices(fra_words, fra_vocab)

            df['uid'].append(index)
            df['source'].append(','.join(eng_indices))
            df['target'].append(','.join(fra_indices))

            # get the glove embedding
            src_tensor = []
            for word in eng_words:
                if word == EOS_token:
                    embedding = EOS_vector
                else:
                    embedding = glove_dict.get(word, None)
                    if embedding is None :
                        embedding = UNK_vector
                        print("{} has no embedding! Sentence: {}".format(word, eng_words))
                embedding = embedding.view(1, -1)
                src_tensor.append(embedding)
            #

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
    df.to_csv(pd_path, index=False)

if __name__ == '__main__':

    eng_vocab_path = '/Users/pjs/byte_play/data/eng_fra/small_eng_vocab.pkl'
    fra_vocab_path = '/Users/pjs/byte_play/data/eng_fra/small_fra_vocab.pkl'

    #create_vocab(eng_vocab_path, fra_vocab_path)

    glove_dict = glove_dict_get()
    eng_vocab = pickle.load(open(eng_vocab_path, 'rb'))
    fra_vocab = pickle.load(open(fra_vocab_path, 'rb'))
    txt_path = '/Users/pjs/byte_play/data/eng_fra/eng-fra-small.txt'
    save_x_y_npy(eng_vocab, fra_vocab, txt_path, glove_dict)
