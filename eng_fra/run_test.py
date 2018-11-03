
import pandas as pd
import sys
sys.path.append("..")
from funcs.gen import EnFraDataSet
from torch.utils.data import DataLoader
from funcs.eval_predict import bleu_compute
from funcs.eval_predict import rogue_compute
from funcs.eval_predict import predict_on_test
from utils.helpers import seq_max_length_get

from easydict import EasyDict as edict
import numpy as np
import pickle
import torch

def predict():

    # load models
    device = torch.device("cpu")
    encoder_path = '/Users/pjs/byte_play/model_pkls/eng_fra_encoder.pkl'
    decoder_path = '/Users/pjs/byte_play/model_pkls/eng_fra_decoder.pkl'
    encoder = torch.load(encoder_path)
    decoder = torch.load(decoder_path)
    print("Load encoder from {}, decoder from {}".format(encoder_path, decoder_path))
    #

    seq_csv_path = '/Users/pjs/byte_play/data/eng_fra/test_small_seq.csv'
    en_vocab_path = '/Users/pjs/byte_play/data/eng_fra/small_eng_vocab.pkl'
    fra_vocab_path = '/Users/pjs/byte_play/data/eng_fra/small_fra_vocab.pkl'

    # Split train / val, TODO
    X = pd.read_csv(seq_csv_path)['source'].values
    Y = pd.read_csv(seq_csv_path)['target'].values
    uids = pd.read_csv(seq_csv_path)['uid'].values


    # read VOCAB
    en_vocab = pickle.load(open(en_vocab_path, 'rb'))
    fra_vocab = pickle.load(open(fra_vocab_path, 'rb'))

    src_vocab_len = len(en_vocab)
    # src_EOS_token = int(en_vocab.index('<EOS>'))
    src_pad_token = int(en_vocab.index('<PAD>'))

    target_vocab_len = len(fra_vocab)
    target_SOS_token = int(fra_vocab.index('<SOS>'))
    target_EOS_token = int(fra_vocab.index('<EOS>'))
    target_pad_token = int(fra_vocab.index('<PAD>'))

    config = edict()
    config.src_vocab_len = src_vocab_len
    config.target_vocab_len = target_vocab_len
    config.src_pad_token = src_pad_token
    config.target_SOS_token = target_SOS_token
    config.target_EOS_token = target_EOS_token
    config.target_pad_token = target_pad_token
    config.use_pretrain_embedding = False
    config.num_workers = 8
    config.encoder_pad_shape = (seq_max_length_get(seq_csv_path, 'source'), 1)
    config.decoder_pad_shape = (seq_max_length_get(seq_csv_path, 'target'), 1)



    # get generator
    test_generator = EnFraDataSet(X, Y, uids, config.encoder_pad_shape, config.decoder_pad_shape,
                                   src_pad_token, target_pad_token, config.use_pretrain_embedding)
    test_loader = DataLoader(test_generator,
                              batch_size=1,
                              shuffle=False,
                              num_workers=config.num_workers,
                              # pin_memory=True
                              )

    # load encoder, decoder

    test_loss = []
    rogues = []
    bleus = []

    for i, (src_tensor, target_tensor, uid) in enumerate(test_loader):
        loss, decoded_words, target_words = predict_on_test(encoder, decoder, src_tensor, target_tensor, fra_vocab,
                                                            device,
                                                            config.target_SOS_token, config.target_pad_token,
                                                            config.src_pad_token,
                                                            EOS_token=config.target_EOS_token,
                                                            teacher_forcing=False,
                                                            batch_size=1)  # TODO, set True only for debug

        print("-----------------------------------------------------")
        print("loss: ", loss)
        print("target_words: ", target_words)
        print("Decoded_words: ", decoded_words)

        # TODO, add language model
        # target_words = eval(y_df[(y_df.id == val_id)]['index'].values[0])
        #

        # compute rogue & bleu
        rogue = rogue_compute(target_words, decoded_words)
        bleu = bleu_compute(target_words, decoded_words)
        #
        #
        test_loss.append(loss)
        rogues.append(rogue)
        bleus.append(bleu)

    print("test_loss: ", np.average(test_loss))
    print("rogues: ", np.average(rogues))
    print("bleus: ", np.average(bleus))


if __name__ == '__main__':
    predict()