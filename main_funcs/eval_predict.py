"""
install rouge: pip install git+https://github.com/tagucci/pythonrouge.git
"""

import torch.nn as nn
import torch
import ipdb
import math
import nltk
import sys
from .lcs import lcsubstring_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------------------------------------------------
# evaluation metrics
# ----------------------------------------------------------------------------------------------------------------------
def bleu_compute(reference, summary):
    # compute BLEU score
    bleu_score = nltk.translate.bleu_score.sentence_bleu([reference], summary)
    #
    return bleu_score


def rogue_compute(reference, summary):
    # compute lcs score
    lcs_len = lcsubstring_length(summary, reference)
    lcs_recall = lcs_len / len(reference)
    lcs_precision = lcs_len / len(summary)
    beta = lcs_precision / (lcs_recall + math.e ** -12)
    rogue_l = ((1 + beta ** 2) * lcs_recall * lcs_precision) / (lcs_recall + beta ** 2 * lcs_precision)
    return rogue_l


# ----------------------------------------------------------------------------------------------------------------------


def _encode(encoder, src_tensors):
    encoder_h0 = encoder.initHidden(1)
    src_tensors = src_tensors.view(1, src_tensors.shape[0], -1)
    encoder_outputs, encoder_hidden = encoder(src_tensors, encoder_h0)

    # encoder_outputs:  torch.Size([150, 1, 256]) encoder_hidden:   torch.Size([1, 1, 256])

    # # encoding
    # encoder_hidden = encoder.initHidden(batch=1)
    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    #
    # for ei in range(input_length):
    #     encoder_output, encoder_hidden = encoder(src_tensors[ei], encoder_hidden)
    #     encoder_outputs[ei] += encoder_output[0, 0]
    # #

    return encoder_outputs, encoder_hidden


def _decode(decoder, encoder_outputs, encoder_hidden, EOS_token, target_tensor=None):

    max_length = len(target_tensor)
    # # decoding
    # if target_tensor is not None:
    #     decoder_input = target_tensor.view(target_tensor.shape[1], target_tensor.shape[0], -1)  # SOS

    decoder_hidden = encoder_hidden
    decoded_outputs = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # initialize decoder_input_t
    decoder_input_t = torch.tensor([[target_tensor[0]]])

    for t in range(max_length):

        # ipdb > decoder_hidden.shape
        # torch.Size([1, 1, 256])
        # ipdb > encoder_outputs.shape
        # torch.Size([150, 1, 256])
        # ipdb > decoder_input_t
        # tensor([[30211]])

        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input_t, decoder_hidden, encoder_outputs)

        decoder_attentions[t] = decoder_attention.data
        decoder_output = decoder_output.squeeze(0)

        _, topi = decoder_output.topk(1)
        decoder_input_t = topi.detach()  # detach from history as input

        #decoder_input_t = target_tensor[t-1].unsqueeze(0)
        #ipdb.set_trace()

        decoded_outputs.append(decoder_output)

        # if int(topi.data[0][0]) == EOS_token:
        #     break
    #

    return decoded_outputs, decoder_attentions, t


def _decode_predict_index(decoded_outputs, vocab):
    # decoded_outputs -> words
    decoded_words = []
    for decoder_output in decoded_outputs:
        _, topi = decoder_output.data.topk(1)
        word_index = int(topi[0][0])
        word = vocab[word_index]
        decoded_words.append(word)
        if word == '<EOS>':
            break
    #
    return decoded_words


def _decode_target_index(target_tensors, vocab):
    target_words = []
    for tensor in target_tensors:
        word_index = int(tensor[0])
        word = vocab[word_index]
        target_words.append(word)
        if word == '<EOS>':
            break
    #
    return target_words


def loss_compute(target_tensors, decoded_outputs):
    # compute loss
    criterion = nn.NLLLoss()
    loss = 0

    for i, decoded_output in enumerate(decoded_outputs):
        gt_output = target_tensors[i]
        loss += criterion(decoded_output, gt_output)

    loss = float(loss / len(target_tensors))

    return loss


def predict(encoder, decoder, src_t_tensor, vocab, max_length=20, SOS_token=0):
    with torch.no_grad():
        # encode
        encoder_outputs, encoder_hidden = _encode(encoder, src_t_tensor)

        # decode
        decoded_outputs, decoder_attentions, di = _decode(decoder, encoder_outputs, encoder_hidden, SOS_token,
                                                          max_length)

        # decode word index into words
        decoded_words = _decode_predict_index(decoded_outputs, vocab)

        return decoded_words


def evaluate(encoder, decoder, src_tensor, target_tensor, vocab, max_length=20, EOS_token=0):
    with torch.no_grad():

        # encode
        encoder_outputs, encoder_hidden = _encode(encoder, src_tensor)

        # decode
        decoded_outputs, decoder_attentions, di = _decode(decoder, encoder_outputs, encoder_hidden, EOS_token,
                                                          target_tensor)

        # compute loss
        loss = loss_compute(target_tensor, decoded_outputs)

        # decode predict word index into words
        decoded_words = _decode_predict_index(decoded_outputs, vocab)

        # decoded_outputs -> words
        target_words = _decode_target_index(target_tensor, vocab)

        return loss, decoded_words, target_words, decoder_attentions[:di + 1]
