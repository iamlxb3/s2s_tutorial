import numpy as np
import torch
import re
import math

# from nltk.tokenize import word_tokenize
# def sentence_to_tensor(sentence, elmo, batch_to_ids):
#     words = word_tokenize(sentence, language='english')
#     eng_ids = batch_to_ids([words])
#     embeddings = elmo(eng_ids)
#     tensor = embeddings['elmo_representations'][0][0]
#     return tensor
#
# def french_sentence_to_int_tensors(sentence, vocab):
#     words = word_tokenize(sentence, language='french')
#
#     tensor_output = np.zeros((len(words), 1))
#
#     for i, word in enumerate(words):
#         word_index = vocab.index(word)
#         tensor_output[i] = word_index
#
#     tensor_output = torch.from_numpy(tensor_output)
#     tensor_output = tensor_output.type(torch.LongTensor)
#     return tensor_output


def init_emlo():
    from allennlp.modules.elmo import Elmo, batch_to_ids

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway" \
                   "/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway" \
                  "/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    return elmo, batch_to_ids

def sort_test_xpaths(x_paths):
    x_paths = sorted(x_paths) # TODO, sort
    x_paths_with_id = []
    for x_path in x_paths:
        id = int(re.findall(r'x_([0-9]+).pt', x_path)[0])
        x_paths_with_id.append((id, x_path))
    x_paths = [x[1] for x in sorted(x_paths_with_id, key=lambda x:x[0])]
    return x_paths

def pos_encode(pos, dim):
    pos_embeddings = []
    for i in range(dim):
        if i % 2 == 0:
            pos_embedding = math.cos(pos / 1000 ** (2 * i / dim))
        else:
            pos_embedding = math.sin(pos / 1000 ** (2 * i / dim))
        pos_embeddings.append(pos_embedding)
    pos_embeddings = np.array([pos_embeddings])
    return pos_embeddings
