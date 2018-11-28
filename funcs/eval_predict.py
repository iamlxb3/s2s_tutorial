"""
install rouge: pip install git+https://github.com/tagucci/pythonrouge.git
"""

import torch
import ipdb
import math
import nltk

import sys

sys.path.append('..')
from utils.helpers import lcsubstring_length
from utils.helpers import encode_func
from utils.helpers import decode_func
from utils.helpers import beam_search

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
def _decode_predict_index(cfg, decoded_outputs, vocab):
    decoded_words = []

    if cfg.decode_mode == 'greedy':
        # decoded_outputs -> words
        for decoder_output in decoded_outputs:
            _, topi = decoder_output.data.topk(1)
            word_index = int(topi[0][0])
            word = vocab[word_index]
            decoded_words.append(word)
            if word == '<EOS>':
                break
        #
    elif cfg.decode_mode == 'beam_search':
        best_seq = beam_search(1, decoded_outputs, cfg.beam_width, return_last=False).view(-1)
        for index in best_seq:
            word = vocab[index]
            decoded_words.append(word)
            if word == '<EOS>':
                break

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


def predict_on_test(cfg, encoder, decoder, src_tensor, target_tensor):
    vocab = cfg.target_vocab
    with torch.no_grad():
        # encode
        encoder_outputs, encoder_last_hidden = encode_func(cfg, src_tensor, encoder)

        # decode
        loss = 0
        loss, target_max_len, decoded_outputs, attn_weights = decode_func(cfg, loss, target_tensor, encoder_outputs,
                                                                          encoder_last_hidden, False,
                                                                          decoder, is_test=True)
        # decode predict word index into words
        decoded_words = _decode_predict_index(cfg, decoded_outputs, vocab)

        # decoded_outputs -> words
        target_words = _decode_target_index(target_tensor, vocab)

        # attn_weights
        attn_weights = [x.view(1, -1) for x in attn_weights]

        return float(loss), decoded_words, target_words, attn_weights


def eval_on_val(cfg, encoder, decoder, src_tensor, target_tensor, use_teacher_forcing=False):
    with torch.no_grad():
        # encode
        encoder_outputs, encoder_last_hidden = encode_func(cfg, src_tensor, encoder)

        # decode
        loss = 0
        loss, target_max_len = decode_func(cfg, loss, target_tensor, encoder_outputs, encoder_last_hidden,
                                           use_teacher_forcing, decoder)

        return float(loss)
