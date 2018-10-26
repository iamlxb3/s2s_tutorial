import numpy as np
import torch
import random
import ipdb
import glob
import os
import pickle
import sys
import pandas as pd
import argparse
sys.path.append("..")

from main_funcs.rnn_encoder import EncoderGru
from main_funcs.attention_decoder import AttnDecoderRNN
from main_funcs.trainer import epoches_train
from main_funcs.gen import torch_random_train_gen
from main_funcs.gen import torch_val_gen
from main_funcs.gen import EnFraDataSet
from main_funcs.eval_predict import evaluate
from main_funcs.eval_predict import bleu_compute
from main_funcs.eval_predict import rogue_compute
from torch.utils.data import DataLoader

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', action="store_true", help='Epoch size', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # TODO, add parsing

    # model config
    load_model = False
    hidden_size = 256
    encoder_nlayers = 1
    input_dim = 300
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
    EOS_index = int(vocab.index('<EOS>'))
    SOS_index = int(vocab.index('<SOS>'))
    ignore_index = Vocab_len

    print("Vocab size: {}, ignore_index: {}".format(Vocab_len, ignore_index))
    print("EOS_index: {}, SOS_index: {}".format(EOS_index, SOS_index))

    use_teacher_forcing = True

    N = 10000
    epoches = 28
    batch_size = 128
    max_length = 82
    num_workers = 8
    lr = 1e-3

    input_shape = (max_length, input_dim)
    output_shape = (max_length, 1)

    #

    # create model
    if load_model:
        encoder1 = torch.load(encoder_path).to(device)
        print("Load encoder from {}.".format(encoder_path))
        attn_decoder1 = torch.load(decoder_path).to(device)
        print("Load decoder from {}.".format(decoder_path))
    else:
        encoder1 = EncoderGru(input_dim, hidden_size, encoder_nlayers).to(device)
        attn_decoder1 = AttnDecoderRNN(hidden_size, Vocab_len, dropout_p=0.1, max_length=max_length).to(device)
        print('model initialization ok!')
    #

    # eval
    encoder1.eval()
    attn_decoder1.eval()
    #

    # get generator
    x_paths = glob.glob(os.path.join(train_x_dir, '*.pt'))

    print("Total: ", len(x_paths))
    x_paths = x_paths[:N]


    random.seed(1) # TODO, add shuffle
    random.shuffle(x_paths)

    val_percent = 0.2
    val_index = int(N * 0.2)
    train_x_paths = x_paths[val_index:N]
    val_x_paths = x_paths[0:val_index]


    # temp
    print("train_x_paths: ", train_x_paths[0:10])
    #

    train_generator = EnFraDataSet(train_x_paths, y_csv_path, input_shape, output_shape, ignore_index)
    val_generator = EnFraDataSet(val_x_paths, y_csv_path, input_shape, output_shape, ignore_index)

    train_loader = DataLoader(train_generator,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              # pin_memory=True
                              )
    val_loader = DataLoader(val_generator,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            # pin_memory=True
                            )
    # train_generator = torch_random_train_gen(train_x_paths, y_csv_path)
    # val_generator = torch_val_gen(val_x_paths, y_csv_path)

    step_size = int(N / batch_size)
    #

    # start training
    epoches_train(train_loader, encoder1, attn_decoder1, epoches, step_size, EOS_index, SOS_index, ignore_index,
                  learning_rate=lr, verbose=True, use_teacher_forcing=use_teacher_forcing, device=device)
    print('training ok!')
    #

    if not load_model:
        # save model
        torch.save(encoder1, encoder_path)
        print("Save encoder to {}.".format(encoder_path))
        torch.save(attn_decoder1, decoder_path)
        print("Save decoder to {}.".format(decoder_path))
        #

    # TODO, add batch for validation
    sys.exit()
    # save model

    # eval
    y_df = pd.read_csv(y_csv_path)
    val_loss = []
    rogues = []
    bleus = []
    for i, src_tensor, target_tensor in enumerate(val_generator):
        val_id = int(re.findall(r'x_([0-9]+).pt', val_x_paths[i])[0])

        loss, decoded_words, target_words, attentions = evaluate(encoder1, attn_decoder1, src_tensor, target_tensor,
                                                                 vocab, max_length=max_length, SOS_token=SOS_index)
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
