"""
install rouge: pip install git+https://github.com/tagucci/pythonrouge.git
"""

import torch.nn as nn
import torch
import ipdb
import math
import nltk

import sys

sys.path.append('..')
from utils.helpers import _sort_batch_seq, lcsubstring_length

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
    try:
        rogue_l = ((1 + beta ** 2) * lcs_recall * lcs_precision) / (lcs_recall + beta ** 2 * lcs_precision)
    except ZeroDivisionError:
        rogue_l = 0.0  # TODO
    return rogue_l


# ----------------------------------------------------------------------------------------------------------------------


def _encode(encoder, src_tensors, device, src_pad_token, batch_size):
    batch_size = src_tensors.shape[0]
    encoder_h0 = encoder.initHidden(batch_size, device)

    # TODO, add pad sequence
    # get the actual length of sequence for each sample, sort by decreasing order
    src_tensors, sorted_seq_lens, sorted_indices = _sort_batch_seq(src_tensors, batch_size, src_pad_token)
    src_tensors = torch.transpose(src_tensors, 0, 1)  # transpose, batch second
    #
    encoder_outputs, encoder_hidden = encoder(src_tensors, sorted_seq_lens, sorted_indices, encoder_h0)

    return encoder_outputs, encoder_hidden


def _decode(cfg, decoder, encoder_outputs, encoder_hidden, SOS_token, target_tensor=None, teacher_forcing=False):
    max_length = target_tensor.size(1)

    decoded_outputs = []

    # initialize decoder_input_t
    decoder_input_t = (torch.ones((target_tensor.size(0), 1)) * SOS_token).long().to(device)
    decoder_hidden = encoder_hidden

    for t in range(max_length):

        if cfg.model_type == 'basic_rnn':
            decoder_output, decoder_hidden = decoder(decoder_input_t, decoder_hidden)
        elif cfg.model_type == 'basic_attn':
            decoder_output, decoder_hidden = decoder(decoder_input_t, decoder_hidden, encoder_outputs)

        decoder_output = decoder_output.squeeze(0)

        if teacher_forcing:
            decoder_input_t = target_tensor[:, t, :]
        else:
            _, topi = decoder_output.topk(1)
            decoder_input_t = topi.detach()  # detach from history as input

        decoded_outputs.append(decoder_output)

    return decoded_outputs


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

    length = target_tensors.size(1)

    for i in range(length):
        word_index = int(target_tensors[:, i, :])
        word = vocab[word_index]
        target_words.append(word)
        if word == '<EOS>':
            break
    return target_words


def loss_compute(target_tensors, decoded_outputs, ignore_index):
    # compute loss
    criterion = nn.NLLLoss(ignore_index=ignore_index)
    loss = 0

    for i, decoded_output in enumerate(decoded_outputs):
        gt_output = target_tensors[:, i, :].squeeze(1)
        loss += criterion(decoded_output, gt_output)

    loss = float(loss / target_tensors.size(1))  # not exact loss, approximate

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


def predict_on_test(encoder, decoder, src_tensor, target_tensor, vocab, device, target_SOS_token, target_pad_token,
                    src_pad_token, EOS_token=0,
                    teacher_forcing=False, batch_size=1):
    with torch.no_grad():
        # encode
        encoder_outputs, encoder_hidden = _encode(encoder, src_tensor, device, src_pad_token, batch_size)

        # decode
        decoded_outputs = _decode(decoder, encoder_outputs, encoder_hidden, target_SOS_token,
                                  target_tensor, teacher_forcing=teacher_forcing)

        # compute loss
        loss = loss_compute(target_tensor, decoded_outputs, target_pad_token)

        # decode predict word index into words
        decoded_words = _decode_predict_index(decoded_outputs, vocab)

        # decoded_outputs -> words
        target_words = _decode_target_index(target_tensor, vocab)

        return loss, decoded_words, target_words


def eval_on_val(cfg, encoder, decoder, src_tensor, target_tensor, device, target_SOS_token, target_pad_token,
                src_pad_token, teacher_forcing=False, batch_size=None):
    with torch.no_grad():
        # encode
        encoder_outputs, encoder_hidden = _encode(encoder, src_tensor, device, src_pad_token, batch_size)

        # decode
        decoded_outputs = _decode(cfg, decoder, encoder_outputs, encoder_hidden, target_SOS_token,
                                  target_tensor, teacher_forcing=teacher_forcing)

        # compute loss
        loss = loss_compute(target_tensor, decoded_outputs, target_pad_token)

        return loss
