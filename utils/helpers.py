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


def _actual_seq_length_compute(input_tensor, batch_size, src_pad_token):
    seq_lens = []
    indices = []
    for batch_i in range(batch_size):
        seq_len = input_tensor[0].squeeze(1)
        seq_len = len([x for x in seq_len if x != src_pad_token])
        seq_lens.append(seq_len)
        indices.append(batch_i)
    #
    return seq_lens, indices


def _sort_batch_seq(input_tensor, batch_size, src_pad_token):
    """

    :param input_tensor: torch.Size([batch_size, seq_max_len, 1])
    :param batch_size:
    :param src_pad_token:
    :return:
    """
    # get the actual length of sequence for each sample
    src_seq_lens, src_seq_indices = _actual_seq_length_compute(input_tensor, batch_size, src_pad_token)
    #

    # sort by decreasing order
    input_tensor_len = sorted(list(zip(input_tensor, src_seq_lens, src_seq_indices)), key=lambda x: x[1], reverse=True)
    input_tensor = torch.cat([x[0].view(1, x[0].size(0), -1) for x in input_tensor_len], 0)
    sorted_seq_lens = [x[1] for x in input_tensor_len]
    sorted_indices = [x[2] for x in input_tensor_len]
    #
    return input_tensor, sorted_seq_lens, sorted_indices

def save_cktpoint(encoder, decoder, encoder_path, decoder_path):
    # save model
    torch.save(encoder, encoder_path)
    print("Save encoder to {}.".format(encoder_path))
    torch.save(decoder, decoder_path)
    print("Save decoder to {}.".format(decoder_path))