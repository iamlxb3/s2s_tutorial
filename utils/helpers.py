import torch
import sys
import pandas as pd

sys.path.append("..")

from main_funcs.encoder import EncoderRnn
from main_funcs.decoder import DecoderRnn


# def model_get(device, load_model, encoder_path, decoder_path, input_dim, hidden_size, encoder_nlayers, target_vocab_len,
#               max_length, is_train=True):
#     # create model
#     if load_model:
#         encoder = torch.load(encoder_path).to(device)
#         print("Load encoder from {}.".format(encoder_path))
#         decoder = torch.load(decoder_path).to(device)
#         print("Load decoder from {}.".format(decoder_path))
#     else:
#         encoder = EncoderGru(input_dim, hidden_size, encoder_nlayers).to(device)
#         decoder = AttnDecoderRNN(hidden_size, target_vocab_len, dropout_p=0.1, max_length=max_length).to(device)
#         print('model initialization ok!')
#     #
#     if is_train:
#         encoder, decoder = encoder.train(), decoder.train()
#     return encoder, decoder


def model_get(device, load_model, encoder_path, decoder_path, config_dict, is_train=True):
    # create model
    if load_model:
        encoder = torch.load(encoder_path).to(device)
        print("Load encoder from {}.".format(encoder_path))
        decoder = torch.load(decoder_path).to(device)
        print("Load decoder from {}.".format(decoder_path))
    else:

        # load encoder config
        encoder_input_dim = config_dict['encoder_input_dim']
        encoder_hidden_dim = config_dict['encoder_hidden_dim']
        src_vocab_len = config_dict['src_vocab_len']
        #

        # load decoder config
        decoder_input_dim = config_dict['decoder_input_dim']
        decoder_hidden_dim = config_dict['decoder_hidden_dim']
        target_vocab_len = config_dict['target_vocab_len']
        #

        encoder = EncoderRnn(encoder_input_dim, encoder_hidden_dim, src_vocab_len).to(device)
        decoder = DecoderRnn(decoder_input_dim, decoder_hidden_dim, target_vocab_len).to(device)
        print('model initialization ok!')
    #
    if is_train:
        encoder, decoder = encoder.train(), decoder.train()
    else:
        encoder, decoder = encoder.eval(), decoder.eval()
    return encoder, decoder

def seq_max_length_get(seq_csv_path, key):
    df = pd.read_csv(seq_csv_path)
    sequences = df[key].values
    max_len = max([len(x.split(',')) for x in sequences])
    return max_len