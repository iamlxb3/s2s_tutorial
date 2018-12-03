import os
import torch
import ipdb
import numpy as np

def glove_dict_get():
    glove_txt_path = '/Users/pjs/byte_play_main/embeddings/glove.840B.300d.txt'
    glove_dict = {}
    with open (glove_txt_path, 'r') as f:
        for i, line in enumerate(f):
            line_split = line.split(' ')
            word = line_split[0]
            embedding = np.array([float(x) for x in line_split[1:]])
            glove_dict[word] = embedding

            # if i == 100000:  # 100000
            #     break

    print("read glove_dict complete!, Size: {}".format(len(glove_dict.keys())))
    return glove_dict # 2196016

def main(glove_dict, save_dir, txt_path):
    content_words = []
    index = 0
    with open(txt_path, 'r') as f:
        for line in f:
            if line.strip() and '-DOCSTART-' not in line:
                word = line.strip().split(' ')[0]
                content_words.append(word)
            else:
                if content_words:
                    src_tensor = []
                    for word in content_words:
                        embedding = glove_dict.get(word, None)
                        if embedding is None:
                            embedding = glove_dict['unk']
                            print("{} has no embedding!".format(word))
                        src_tensor.append(np.expand_dims(embedding, 0))

                    src_tensor = np.concatenate(src_tensor, axis=0)
                    x_save_path = os.path.join(save_dir, 'x_{}.npy'.format(index))
                    np.save(x_save_path, src_tensor)
                    index += 1

                content_words = []




if __name__ == '__main__':
    glove_dict = glove_dict_get()
    train_save_dir = '/Users/pjs/byte_play_main/data/CoNLL_2003/train_embed'
    test_save_dir = '/Users/pjs/byte_play_main/data/CoNLL_2003/test_embed'
    train_txt_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/raw2/train.txt'
    test_txt_path = '/Users/pjs/byte_play_main/data/CoNLL_2003/raw2/test.txt'

    main(glove_dict, train_save_dir, train_txt_path)
    main(glove_dict, test_save_dir, test_txt_path)

