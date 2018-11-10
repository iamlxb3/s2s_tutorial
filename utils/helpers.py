import torch
import math
import ipdb
import operator
import numpy as np
import sys
import pandas as pd

sys.path.append("..")

from funcs.encoder import Encoder
from funcs.decoder import DecoderRnn
from funcs.decoder import AttnDecoderRNN
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from queue import PriorityQueue


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

        encoder = Encoder(encoder_input_dim, encoder_hidden_dim, src_vocab_len,
                          bidirectional=cfg.encoder_bi_direction, type=cfg.rnn_type).to(cfg.device)
        if cfg.model_type == 'basic_rnn':
            decoder = DecoderRnn(decoder_input_dim, decoder_hidden_dim, target_vocab_len).to(cfg.device)
        elif cfg.model_type == 'basic_attn':
            decoder = AttnDecoderRNN(cfg.attn_method, target_vocab_len, decoder_input_dim, decoder_hidden_dim,
                                     softmax_share_embedd=cfg.softmax_share_embedd,
                                     is_point_generator=cfg.is_point_generator,
                                     pad_token=cfg.target_pad_token).to(cfg.device)

        # share embedding
        if cfg.share_embedding:
            decoder.embedding = encoder.embedding
        #

        print('model initialization ok!')
    #

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
        seq_len = input_tensor[batch_i].squeeze(1)
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


def encode_func(cfg, input_tensor, encoder):
    batch_size = input_tensor.shape[0]

    encoder_h0 = encoder.initHidden(batch_size, cfg.device)

    # get the actual length of sequence for each sample, sort by decreasing order
    input_tensor, sorted_seq_lens, sorted_indices = _sort_batch_seq(input_tensor, batch_size, cfg.src_pad_token)
    input_tensor = torch.transpose(input_tensor, 0, 1)  # transpose, batch second
    #

    # encode
    encoder_outputs, encoder_hidden = encoder(input_tensor, sorted_seq_lens, sorted_indices, encoder_h0)
    #

    return encoder_outputs, encoder_hidden


def greedy_decode(cfg, decoder, encoder_outputs, target_tensor, encoder_last_hidden, target_max_len,
                  use_teacher_forcing,
                  coverage_loss=None,
                  coverage_vector=None,
                  coverage_mask_list=None,
                  word_pointer_pos=None):
    # initialize decoder_hidden, decoder_input_0
    decoder_hidden = encoder_last_hidden
    decoder_input_t = (torch.ones((target_tensor.size(0), 1)) * cfg.target_SOS_token).long().to(cfg.device)

    attn_weights = []
    decoder_output = []

    for t in range(target_max_len):

        # (1.) basic rnn
        if cfg.model_type == 'basic_rnn':
            decoder_output_t, decoder_hidden = decoder(decoder_input_t, decoder_hidden)
        # (2.) basic attn
        elif cfg.model_type == 'basic_attn':
            # coverage
            if cfg.is_coverage:
                if t != 0:
                    coverage_mask = coverage_mask_list[t]
                    coverage_loss_t = torch.sum(torch.min(coverage_vector, attn_weight_t.squeeze(1)), 1)
                    coverage_loss_t = coverage_loss_t[coverage_mask]
                    coverage_loss_t = torch.sum(coverage_loss_t)
                    coverage_vector = coverage_vector + attn_weight_t.squeeze(1)
                elif t == 0:
                    coverage_loss_t = 0
                    # print("{}-coverage_loss: {}".format(t, coverage_loss_t))
            decoder_output_t, decoder_hidden, attn_weight_t = decoder(decoder_input_t, decoder_hidden, encoder_outputs,
                                                                      coverage=coverage_vector,
                                                                      word_pointer_pos=word_pointer_pos)
            attn_weights.append(attn_weight_t)

        if not use_teacher_forcing:
            topv, topi = decoder_output_t.topk(1)
            decoder_input_t = topi.detach()  # .detach() or not?  # detach from history as input
        else:
            decoder_input_t = target_tensor[:, t, :]

        # accumulate coverage loss
        if cfg.is_coverage:
            coverage_loss += coverage_loss_t
        #

        # gather decoder output
        decoder_output.append(decoder_output_t)
        #
    return decoder_output, coverage_loss, attn_weights


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(cfg, target_tensor, encoder_last_hidden, encoder_outputs):
    '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''

    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []
    decoder_hiddens = encoder_last_hidden

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input_t = (torch.ones((target_tensor.size(0), 1)) * cfg.target_SOS_token).long().to(cfg.evice)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input_t, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == cfg.target_EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch


def decode_func(cfg, loss, target_tensor, encoder_outputs, encoder_last_hidden, use_teacher_forcing, decoder,
                is_test=False, input_tensor=None):
    # load config
    verbose = cfg.verbose
    target_pad_token = cfg.target_pad_token

    device = cfg.device
    target_max_len = target_tensor.size(1)
    criterion = cfg.criterion

    # add coverage
    coverage_vector = None
    if cfg.is_coverage:
        coverage_loss = 0
        coverage_vector = torch.zeros(encoder_outputs.size(1), encoder_outputs.size(0))
        # coverage mask
        coverage_mask_list = []
        for t in range(target_max_len):
            mask = target_tensor[:, t, :] != target_pad_token
            coverage_mask_list.append(mask.view(-1))
        #
    #

    # point-generator
    word_pointer_pos = None
    if cfg.is_point_generator:
        batch_size = cfg.batch_size
        word_pointer_pos = torch.zeros((batch_size, cfg.vocab_size, encoder_outputs.size(0)))
        input_tensors = []
        for batch in range(batch_size):
            batch_tensor = input_tensor[batch][input_tensor[batch] != cfg.src_pad_token].unsqueeze(0)
            input_tensors.append(batch_tensor)

        for word_index in range(cfg.vocab_size):
            for batch in range(batch_size):
                input_tensor = input_tensors[batch]
                word_pointer_pos[batch][word_index][:input_tensor.size(1)] = input_tensor == word_index
    #

    # decode
    if cfg.decode_mode == 'greedy':
        decoder_output, coverage_loss, attn_weights = greedy_decode(cfg, decoder, encoder_outputs, target_tensor,
                                                                    encoder_last_hidden, target_max_len,
                                                                    use_teacher_forcing,
                                                                    coverage_loss=coverage_loss,
                                                                    coverage_vector=coverage_vector,
                                                                    coverage_mask_list=coverage_mask_list,
                                                                    word_pointer_pos=word_pointer_pos)
    #

    # decoder_output, target_tensor,
    loss += criterion(torch.cat(decoder_output, 0), target_tensor.view(-1))
    loss += coverage_loss / target_max_len  # not accurate because of the padding

    # if verbose:
    #     print_target = [int(x) for x in target_tensor[0] if int(x) != target_pad_token]
    #     new_decoded_outputs = []
    #     for x in decoded_outputs:
    #         new_decoded_outputs.append(x)
    #         if x == target_EOS_index:
    #             break
    #
    #     print("\n--------------------------------------------")
    #     print("decoded_outputs: ", new_decoded_outputs)
    #     print("target_tensor: ", print_target)
    #     print("Overlap: ", len(set(print_target).intersection(new_decoded_outputs)) / len(print_target))

    if is_test:
        return loss, target_max_len, decoder_output, attn_weights

    return loss, target_max_len


def plot_attentions(attn_weights, src, target):
    """

    :param attn_weights: M x N , M: length of the target
    :param src:
    :param target:
    :return:
    """
    attn_weights = torch.cat(attn_weights).numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_weights, cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticklabels([''] + src, rotation=90)
    ax.set_yticklabels([''] + target)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
