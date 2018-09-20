import os
import nltk
import glob
import ipdb
import pickle
import collections
from nltk.tokenize import word_tokenize


def main():
    data_dir = '/Users/pjs/byte_play/data/bytecup2018'
    txt_paths = glob.glob(os.path.join(data_dir, '*.txt'))

    bytecup_vocab = []
    for txt_path in txt_paths[0:4]:
        with open (txt_path, 'r') as f:
            for i, line in enumerate(f):
                dict1 = eval(line)
                content = dict1['content']
                title = dict1.get('title', '')

                words1 = word_tokenize(content)
                words1 = [x.lower() for x in words1]
                words2 = word_tokenize(title)
                words2 = [x.lower() for x in words2]
                bytecup_vocab.extend(words1)
                bytecup_vocab.extend(words2)

                if i % 100 == 0:
                    print("Vocab size: {}".format(len(bytecup_vocab)))


            print("Parse {} file done!".format(txt_path))

    # top 80, 000 words

    bytecup_vocab = sorted(collections.Counter(bytecup_vocab).items(), key=lambda x:x[1], reverse=True)
    bytecup_vocab = [x[0] for x in bytecup_vocab][:80000]
    bytecup_vocab = ['<UNK>', '<SOS>', '<EOS>'] + bytecup_vocab
    save_path = 'bytecup_vocab.pkl'
    pickle.dump(bytecup_vocab, open(save_path, 'wb'))
    print("Save to {}".format(save_path))

if __name__ == '__main__':
    main()