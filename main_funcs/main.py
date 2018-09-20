from rnn_encoder import EncoderRNN
from attention_decoder import AttnDecoderRNN
from trainer import trainIters
import numpy as np
import torch
import random
import ipdb
import glob
import os
import pickle
from gen import torch_random_train_gen
from gen import torch_val_gen
from eval import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # model config
    hidden_size = 246
    encoder_nlayers = 1
    input_dim = 1024
    #

    # set path
    data_set = 'eng_fra'
    top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(top_dir, 'data')
    train_x_dir = os.path.join(data_dir, data_set, 'train')
    y_csv_path = os.path.join(data_dir, data_set, 'train_y.csv')
    vocab_path = os.path.join(data_dir, data_set, 'vocab.pkl')
    #

    # training config
    vocab = pickle.load(open(vocab_path, 'rb'))
    Vocab_len = len(vocab)
    EOS_token = int(vocab.index('<EOS>'))
    SOS_token = int(vocab.index('<SOS>'))

    N = 500
    epoches = 10
    batch_size = 1
    Y_max_length = 88
    #

    # create model
    encoder1 = EncoderRNN(input_dim, hidden_size, encoder_nlayers).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, Vocab_len, dropout_p=0.1, max_length=Y_max_length).to(device)
    print('model initialization ok!')
    #

    # get generator
    x_paths = glob.glob(os.path.join(train_x_dir, '*.pt'))[0:N]
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
               Y_max_length=Y_max_length)
    print('training ok!')
    #

    # eval
    val_loss = []
    lcs_ratios = []
    bleus = []
    for src_tensor, target_tensor in val_generator:
        loss, decoded_words, attentions, lcs_ratio, bleu = evaluate(encoder1, attn_decoder1, src_tensor, target_tensor,
                                                                    vocab,
                                                                    max_length=Y_max_length, SOS_token=SOS_token,
                                                                    EOS_token=EOS_token)
        val_loss.append(loss)
        lcs_ratios.append(lcs_ratio)
        bleus.append(bleu)
        print("Decoded_words: ", decoded_words)

    print("val_loss: ", np.average(val_loss))
    print("lcs_ratio: ", np.average(lcs_ratios))
    print("bleu: ", np.average(bleus))

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
