import torch
import sys

sys.path.append("..")

from main_funcs.rnn_encoder import EncoderRNN
from main_funcs.attention_decoder import AttnDecoderRNN


def model_get(device, load_model, encoder_path, decoder_path, input_dim, hidden_size, encoder_nlayers, Vocab_len,
              max_length, is_train=True):
    # create model
    if load_model:
        encoder = torch.load(encoder_path).to(device)
        print("Load encoder from {}.".format(encoder_path))
        decoder = torch.load(decoder_path).to(device)
        print("Load decoder from {}.".format(decoder_path))
    else:
        encoder = EncoderRNN(input_dim, hidden_size, encoder_nlayers).to(device)
        decoder = AttnDecoderRNN(hidden_size, Vocab_len, dropout_p=0.1, max_length=max_length).to(device)
        print('model initialization ok!')
    #
    if is_train:
        encoder, decoder = encoder.train(), decoder.train()
    return encoder, decoder
