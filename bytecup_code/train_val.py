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

from main_funcs.rnn_encoder import EncoderGru
from main_funcs.attention_decoder import AttnDecoderRNN
from main_funcs.trainer import trainIters
from main_funcs.gen import torch_random_train_gen
from main_funcs.gen import torch_val_gen
from main_funcs.eval_predict import eval_on_val
from main_funcs.eval_predict import bleu_compute
from main_funcs.eval_predict import rogue_compute

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

if __name__ == '__main__':
    # model config
    hidden_size = 256
    encoder_nlayers = 1
    input_dim = 1024
    #

    # set path
    data_set = 'bytecup2018'
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data')
    pkl_dir = os.path.join(top_dir, 'model_pkls')
    train_x_dir = os.path.join(data_dir, data_set, 'train')
    y_csv_path = os.path.join(data_dir, data_set, 'train_y.csv')
    vocab_path = os.path.join(data_dir, data_set, 'bytecup_vocab.pkl')
    encoder_path = os.path.join(pkl_dir, '{}_encoder.pkl'.format(data_set))
    decoder_path = os.path.join(pkl_dir, '{}_decoder.pkl'.format(data_set))
    #

    # training config
    vocab = pickle.load(open(vocab_path, 'rb'))
    Vocab_len = len(vocab)
    EOS_token = int(vocab.index('<EOS>'))
    SOS_token = int(vocab.index('<SOS>'))

    N = 20
    epoches = 1
    batch_size = 1
    max_length = 2000
    #

    # create model
    encoder1 = EncoderGru(input_dim, hidden_size, encoder_nlayers).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, Vocab_len, dropout_p=0.1, max_length=max_length).to(device)
    print('model initialization ok!')
    #

    # get generator
    x_paths = glob.glob(os.path.join(train_x_dir, '*.pt'))[0:N]
    random.seed(1)
    random.shuffle(x_paths)
    val_percent = 0.2
    val_index = int(N * 0.2)
    train_x_paths = x_paths[val_index:N]
    val_x_paths = x_paths[0:val_index]
    train_generator = torch_random_train_gen(train_x_paths, y_csv_path)
    val_generator = torch_val_gen(val_x_paths, y_csv_path)

    step_size = int(N / batch_size)
    #

    # start training
    trainIters(train_generator, encoder1, attn_decoder1, epoches, step_size, EOS_token, SOS_token, learning_rate=0.01,
               max_length=max_length, verbose=True)
    print('training ok!')
    #

    # save model

    # eval
    y_df = pd.read_csv(y_csv_path)
    val_loss = []
    rogues = []
    bleus = []
    for src_tensor, target_tensor, id in val_generator:
        loss, decoded_words, target_words, attentions = eval_on_val(encoder1, attn_decoder1, src_tensor, target_tensor,
                                                                    vocab, max_length=max_length, target_SOS_token=SOS_token)
        # TODO, add language model


        target_words = eval(y_df[(y_df.id == id)]['title'].values[0])

        # compute rogue & bleu
        rogue = rogue_compute(target_words, decoded_words)
        bleu = bleu_compute(target_words, decoded_words)
        #

        val_loss.append(loss)
        rogues.append(rogue)
        bleus.append(bleu)

        print("-----------------------------------------------------")
        print("target_words: ", target_words)
        print("Decoded_words: ", decoded_words)

    print("val_loss: ", np.average(val_loss))
    print("val_rogue: ", np.average(rogues))
    print("val_bleu: ", np.average(bleus))

    # save model
    torch.save(encoder1, encoder_path)
    print("Save encoder to {}.".format(encoder_path))
    torch.save(attn_decoder1, decoder_path)
    print("Save decoder to {}.".format(decoder_path))

    #

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
