import nltk
import ipdb
import pickle
import gensim
import os
import numpy as np

words = set()
path = r'D:\KIKI\bytecup2018\bytecup_small.txt'

def V_create():
    # create V
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            dict1 = eval(line)
            content = dict1['content']
            title = dict1['title']
            content_tokens = nltk.word_tokenize(content)
            title_tokens = nltk.word_tokenize(title)
            words.update(set(content_tokens))
            words.update(set(title_tokens))
            print("Size: ", len(words))

    V = sorted(list(words))
    pickle.dump(V, open('V.pkl', 'wb'))


def embedding_create():
    word2vector = 'GoogleNews-vectors-negative300-small.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vector, binary=True)
    V = pickle.load(open('V.pkl', 'rb'))
    save_dir = r'D:\byte_cup\data\baseline\train'

    Y_max_T = 25

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            X = []
            Y = []
            dict1 = eval(line)
            content = dict1['content']
            title = dict1['title']
            content_tokens = nltk.word_tokenize(content)
            title_tokens = nltk.word_tokenize(title)

            for token in content_tokens:
                try:
                    word_vector = model[token]
                except Exception as e:
                    print("token - {}".format(e))
                    continue
                else:
                    word_vector = np.expand_dims(word_vector, axis=0)
                    X.append(word_vector)

            for t in range(Y_max_T):

                y = np.zeros((1, len(V) + 1))

                if t <= len(title_tokens) - 1:
                    token = title_tokens[t]
                    index = V.index(token)
                    y[0][index] = 1.0
                else:
                    y[0][len(V)] = 1.0

                Y.append(y)

            X = np.concatenate(X)
            Y = np.concatenate(Y)

            x_save_path = os.path.join(save_dir, 'x_{}.npy'.format(i))
            y_save_path = os.path.join(save_dir, 'y_{}.npy'.format(i))

            np.save(x_save_path, X)
            np.save(y_save_path, Y)



embedding_create()