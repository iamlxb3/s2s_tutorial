import numpy as np
import pickle
import ipdb

def generate():
    vocab_file = '/Users/pjs/byte_play/embeddings/glove.840B.300d.txt'
    vectors_file = '/Users/pjs/byte_play/embeddings/glove.840B.300d.txt'


    # with open(vocab_file, 'r') as f:
    #     words = [x.rstrip().split(' ')[0] for x in f.readlines()]

    words = []
    with open(vectors_file, 'r') as f:
        vectors = {}
        for i, line in enumerate(f):
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]
            words.append(vals[0])
            if i % 10000 == 0:
                print("{} done!".format(i))

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T

    pickle.dump(vocab, open('/Users/pjs/byte_play/embeddings/glove_vocab.pkl', 'wb'))
    pickle.dump(W_norm, open('/Users/pjs/byte_play/embeddings/glove_W_norm.pkl', 'wb'))


    return (W_norm, vocab, ivocab)


generate()