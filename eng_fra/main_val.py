import numpy as np
import torch
import random
import ipdb
import glob
import os
import pickle
import sys
import pandas as pd

sys.path.append("..")

from main_funcs.rnn_encoder import EncoderRNN
from main_funcs.attention_decoder import AttnDecoderRNN
from main_funcs.gen import EnFraDataSet
from main_funcs.eval_predict import evaluate
from main_funcs.eval_predict import bleu_compute
from main_funcs.eval_predict import rogue_compute
from torch.utils.data import DataLoader
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # model config
    use_train_data = True
    hidden_size = 256
    encoder_nlayers = 1
    input_dim = 1024
    #

    # set path
    data_set = 'eng_fra'
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data')
    pkl_dir = os.path.join(top_dir, 'model_pkls')
    train_x_dir = os.path.join(data_dir, data_set, 'train')
    y_csv_path = os.path.join(data_dir, data_set, 'train_y.csv')
    vocab_path = os.path.join(data_dir, data_set, 'vocab.pkl')
    encoder_path = os.path.join(pkl_dir, '{}_encoder.pkl'.format(data_set))
    decoder_path = os.path.join(pkl_dir, '{}_decoder.pkl'.format(data_set))
    #

    # training config
    vocab = pickle.load(open(vocab_path, 'rb'))
    Vocab_len = len(vocab)
    EOS_token = int(vocab.index('<EOS>'))
    SOS_token = int(vocab.index('<SOS>'))
    ignore_index = 30212

    N = 1000
    epoches = 1
    batch_size = 1
    max_length = 82
    num_workers = 1
    lr = 1e-3

    input_shape = (max_length, input_dim)
    output_shape = (max_length, 1)

    #

    # get generator
    x_paths = glob.glob(os.path.join(train_x_dir, '*.pt'))[0:N]
    # random.seed(1)
    # random.shuffle(x_paths)
    val_percent = 0.2
    val_index = int(N * 0.2)
    train_x_paths = x_paths[val_index:N]
    val_x_paths = x_paths[0:val_index]

    if use_train_data:
        val_x_paths = train_x_paths[0:10]
        print("val_x_paths: ", val_x_paths)

    val_generator = EnFraDataSet(val_x_paths, y_csv_path, input_shape, output_shape,ignore_index=ignore_index)
    assert batch_size == 1
    val_loader = DataLoader(val_generator,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            # pin_memory=True
                            )


    # save model
    encoder1 = torch.load( encoder_path).to(device)
    print("Load encoder from {}.".format(encoder_path))
    attn_decoder1 = torch.load(decoder_path).to(device)
    print("Load decoder from {}.".format(decoder_path))
    #

    # eval
    y_df = pd.read_csv(y_csv_path)
    val_loss = []
    rogues = []
    bleus = []

    for i, (src_tensor, target_tensor) in enumerate(val_generator):
        val_id = int(re.findall(r'x_([0-9]+).pt', val_x_paths[i])[0])

        print("target_tensor: ", target_tensor)

        loss, decoded_words, target_words, attentions = evaluate(encoder1, attn_decoder1, src_tensor, target_tensor,
                                                                 vocab, max_length=max_length, EOS_token=EOS_token)

        print("-----------------------------------------------------")
        print("loss: ", loss)
        print("target_words: ", target_words)
        print("Decoded_words: ", decoded_words)

        # TODO, add language model
        #target_words = eval(y_df[(y_df.id == val_id)]['index'].values[0])
        #

        # compute rogue & bleu
        rogue = rogue_compute(target_words, decoded_words)
        bleu = bleu_compute(target_words, decoded_words)
        #
        #
        val_loss.append(loss)
        rogues.append(rogue)
        bleus.append(bleu)



    print("val_loss: ", np.average(val_loss))
    print("val_rogue: ", np.average(rogues))
    print("val_bleu: ", np.average(bleus))



    # # ------------------------------------------------------------------------------------------------------------------
    # # manual test
    # # ------------------------------------------------------------------------------------------------------------------
    # # init elmo
    # elmo, batch_to_ids = init_emlo()
    # from nltk.tokenize import word_tokenize
    # src_sentence = "I can't be sure."
    # target_sentence = "Je ne saurais en être certain."
    #
    # # src
    # src_tensors = sentence_to_tensor(src_sentence, elmo, batch_to_ids)
    # target_tensors = french_sentence_to_int_tensors(target_sentence, vocab)
    # # ------------------------------------------------------------------------------------------------------------------
