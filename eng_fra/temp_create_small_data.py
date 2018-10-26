import pandas as pd
import unicodedata

eng_prefixes = (
    "i am ", "i'm ",
    "he is", "he's ",
    "she is", "she's",
    "you are", "you're ",
    "we are", "we're ",
    "they are", "they're "
)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def main():
    MAX_LENGTH = 10
    raw_txt_path = '/Users/pjs/byte_play/data/eng_fra/eng-fra.txt'
    new_txt_path = '/Users/pjs/byte_play/data/eng_fra/eng-fra-small.txt'
    with open(raw_txt_path, 'r') as f1:
        with open(new_txt_path, 'w') as f2:
            for line in f1:
                line = line.strip()
                eng, fra = line.split('\t')
                if '-' in eng:
                    continue
                eng = eng.replace('.', ' .')
                eng = eng.replace('!', ' !')
                eng = eng.replace('?', ' ?')
                eng_words, fra_words = eng.split(' '), fra.split(' ')

                fra = fra.lower()
                eng = eng.lower()
                eng = eng.replace("i'm", "i am")
                eng = eng.replace("he's", "he is")
                eng = eng.replace("she's", "she is")
                eng = eng.replace("you're", "you are")
                eng = eng.replace("we're", "we are")
                eng = eng.replace("they're", "they are")
                eng = eng.replace("aren't", "are not")
                eng = eng.replace("isn't", "is not")
                eng = eng.replace("'ll", " will")
                eng = eng.replace("can't", "can not")
                eng = eng.replace("didn't", "did not")
                eng = eng.replace("don't", "do not")
                eng = eng.replace("weren't", "were not")
                eng = eng.replace("doesn't", "does not")
                eng = eng.replace("won't", "will not")
                eng = eng.replace("'ve", " have")
                eng = eng.replace("couldn't", "could not")
                eng = eng.replace("wasn't", "was not")
                eng = eng.replace("haven't", "have not")
                #eng = eng.replace("it's", "it is")

                eng = unicodeToAscii(eng)
                #fra = unicodeToAscii(fra)
                fra = fra.strip()
                fra = fra.replace(' .', '.')
                fra = fra.replace(' !', '!')
                fra = fra.replace(' ?', '?')
                fra = fra.replace('.', ' .')
                fra = fra.replace('!', ' !')
                fra = fra.replace('?', ' ?')

                if len(eng_words) > MAX_LENGTH or len(fra_words) > MAX_LENGTH:
                    continue
                elif not eng.lower().startswith(eng_prefixes):
                    continue
                else:
                    f2.write(eng + '\t' + fra + '\n')


if __name__ == '__main__':
    main()
