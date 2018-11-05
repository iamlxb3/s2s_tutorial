import torch
import math
import numpy as np
import sys
import pandas as pd

sys.path.append("..")

from funcs.encoder import EncoderRnn
from funcs.decoder import DecoderRnn
from funcs.decoder import AttnDecoderRNN

def pos_encode(pos, dim):
    pos_embeddings = []
    for i in range(dim):
        if i % 2 == 0:
            pos_embedding = math.cos(pos / 1000 ** (2 * i / dim))
        else:
            pos_embedding = math.sin(pos / 1000 ** (2 * i / dim))
        pos_embeddings.append(pos_embedding)
    pos_embeddings = np.array([pos_embeddings])
    return pos_embeddings



def model_get(cfg):
    # create model
    if cfg.load_model:
        encoder = torch.load(cfg.encoder_path).to(cfg.device)
        print("Load encoder from {}.".format(cfg.encoder_path))
        decoder = torch.load(cfg.decoder_path).to(cfg.device)
        print("Load decoder from {}.".format(cfg.decoder_path))
    else:

        # load encoder config
        encoder_input_dim = cfg['encoder_input_dim']
        encoder_hidden_dim = cfg['encoder_hidden_dim']
        src_vocab_len = cfg['src_vocab_len']
        #

        # load decoder config
        decoder_input_dim = cfg['decoder_input_dim']
        decoder_hidden_dim = cfg['decoder_hidden_dim']
        target_vocab_len = cfg['target_vocab_len']
        #

        encoder = EncoderRnn(encoder_input_dim, encoder_hidden_dim, src_vocab_len).to(cfg.device)
        if cfg.model_type == 'basic_rnn':
            decoder = DecoderRnn(decoder_input_dim, decoder_hidden_dim, target_vocab_len).to(cfg.device)
        elif cfg.model_type == 'basic_attn':
            decoder = AttnDecoderRNN('general', target_vocab_len, decoder_input_dim, decoder_hidden_dim).to(cfg.device)
        print('model initialization ok!')
    #

    encoder, decoder = encoder.train(), decoder.train()
    # if is_train:
    #     encoder, decoder = encoder.train(), decoder.train()
    # else:
    #     encoder, decoder = encoder.eval(), decoder.eval()
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


def lcsubstring_length(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    l = 0
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            if ca == cb:
                table[i][j] = table[i - 1][j - 1] + 1
                if table[i][j] > l:
                    l = table[i][j]
    return l