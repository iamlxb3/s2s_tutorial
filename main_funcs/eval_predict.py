"""
install rouge: pip install git+https://github.com/tagucci/pythonrouge.git
"""

import torch.nn as nn
import torch
import ipdb
import math
import nltk
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
    rogue_l = ((1 + beta ** 2) *  lcs_recall * lcs_precision) / (lcs_recall + beta ** 2 * lcs_precision)
    return rogue_l
# ----------------------------------------------------------------------------------------------------------------------


def _encode(encoder, src_tensors, max_length):
    input_length = len(src_tensors)

    # encoding
    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(src_tensors[ei], encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]
    #
    return encoder_outputs, encoder_hidden

def _decode(decoder, encoder_outputs, encoder_hidden, SOS_token, max_length):
    # decoding
    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
    decoder_hidden = encoder_hidden
    decoded_outputs = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        decoder_output_t = decoder_output.data
        decoder_input = decoder_output_t.topk(1)[1].squeeze().detach()
        decoded_outputs.append(decoder_output_t)
    #
    return decoded_outputs, decoder_attentions, di

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

    for i, gt_output in enumerate(target_tensors):
        decoded_output = decoded_outputs[i]
        loss += criterion(decoded_output, gt_output)
    loss = float(loss / len(target_tensors))

    return loss

def predict(encoder, decoder, src_t_tensor, vocab, max_length=20, SOS_token=0):
    with torch.no_grad():

        # encode
        encoder_outputs, encoder_hidden = _encode(encoder, src_t_tensor, max_length)

        # decode
        decoded_outputs, decoder_attentions, di = _decode(decoder, encoder_outputs, encoder_hidden, SOS_token, max_length)

        # decode word index into words
        decoded_words = _decode_predict_index(decoded_outputs, vocab)

        return decoded_words

def evaluate(encoder, decoder, src_tensors, target_tensors, vocab, max_length=20, SOS_token=0):
    with torch.no_grad():

        # encode
        encoder_outputs, encoder_hidden = _encode(encoder, src_tensors, max_length)

        # decode
        decoded_outputs, decoder_attentions, di = _decode(decoder, encoder_outputs, encoder_hidden, SOS_token, max_length)

        # compute loss
        loss = loss_compute(target_tensors, decoded_outputs)

        # decode predict word index into words
        decoded_words = _decode_predict_index(decoded_outputs, vocab)

        # decoded_outputs -> words
        target_words = _decode_target_index(target_tensors, vocab)

        return loss, decoded_words, target_words, decoder_attentions[:di + 1]