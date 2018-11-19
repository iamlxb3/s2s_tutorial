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
import seaborn as sns
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


def _decode(cfg, decoder, encoder_outputs, target_tensor, encoder_last_hidden, target_max_len,
            use_teacher_forcing,
            coverage_loss=0,
            coverage_vector=None,
            coverage_mask_list=None,
            is_test=False):
    # initialize decoder_hidden, decoder_input_0
    decoder_hidden = encoder_last_hidden
    decoder_input_t = (torch.ones((target_tensor.size(0), 1)) * cfg.target_SOS_token).long().to(cfg.device)

    attn_weights = []
    decoder_output = []

    loss_ = 0
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
                                                                      coverage=coverage_vector)
            attn_weights.append(attn_weight_t)

        # gather decoder output
        decoder_output.append(decoder_output_t)
        #

        # ipdb> decoder_input_t, torch.Size([4, 1])
        # tensor([[  4],
        #         [  2],
        #         [ 12],
        #         [  2]])

        if not use_teacher_forcing:
            # TODO, optimise beam search
            if cfg.decode_mode == 'beam_search':
                if is_test:
                    batch_size = cfg.test_batch_size
                else:
                    batch_size = cfg.batch_size
                decoder_input_t = beam_search(batch_size, decoder_output, cfg.beam_width)
            elif cfg.decode_mode == 'greedy':
                topv, topi = decoder_output_t.topk(1)
                decoder_input_t = topi.detach()  # .detach() or not?  # detach from history as input
        else:
            decoder_input_t = target_tensor[:, t, :]

        # accumulate coverage loss
        if cfg.is_coverage:
            coverage_loss += coverage_loss_t
        #

    return decoder_output, coverage_loss, attn_weights


# beam search
def beam_search(batch_size, tensor_data, k, return_last=True):
    # TODO, implement beam search in batch
    batch_data = []
    for i in range(batch_size):
        data_1_batch = []
        for tensor_t in tensor_data:
            data_1_batch.append(list(tensor_t[i].detach().numpy()))
        batch_data.append(data_1_batch)

    output = []

    for data in batch_data:
        sequences = [[list(), 1.0]]
        # walk over each step in sequence
        for row in data:
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score * -1 * row[j]]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            # select k best
            sequences = ordered[:k]
        if return_last:
            best_seq = [sorted(sequences, key=lambda x: x[1], reverse=True)[0][0][-1]]
        else:
            best_seq = [sorted(sequences, key=lambda x: x[1], reverse=True)[0][0]]
        output.append(best_seq)

    output = torch.tensor(output)
    return output


def decode_func(cfg, loss, target_tensor, encoder_outputs, encoder_last_hidden, use_teacher_forcing, decoder,
                is_test=False):
    # load config
    verbose = cfg.verbose
    target_pad_token = cfg.target_pad_token

    target_max_len = target_tensor.size(1)
    criterion = cfg.criterion

    # add coverage
    coverage_vector = None
    coverage_mask_list = None
    coverage_loss = 0
    if cfg.is_coverage:
        coverage_vector = torch.zeros(encoder_outputs.size(1), encoder_outputs.size(0))
        # coverage mask
        coverage_mask_list = []
        for t in range(target_max_len):
            mask = target_tensor[:, t, :] != target_pad_token
            coverage_mask_list.append(mask.view(-1))
        #
    #

    # decode
    decoder_output, coverage_loss, attn_weights = _decode(cfg, decoder, encoder_outputs, target_tensor,
                                                          encoder_last_hidden, target_max_len,
                                                          use_teacher_forcing,
                                                          coverage_loss=coverage_loss,
                                                          coverage_vector=coverage_vector,
                                                          coverage_mask_list=coverage_mask_list,
                                                          is_test=is_test)
    #

    # decoder_output, target_tensor,
    loss += criterion(torch.cat(decoder_output, 0), torch.transpose(target_tensor, 0, 1).contiguous().view(-1))
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


def plot_results(epoch_recorder, title='', save_path='', is_show=True):
    """
    Plot the train loss and the validation loss
    :param epoch_recorder:
    :return:
    """
    sns.set_style("darkgrid")

    train_loss = epoch_recorder.train_losses
    val_loss = epoch_recorder.val_losses
    assert len(train_loss) == len(val_loss)
    epoches = [x for x in range(len(train_loss))]

    plt.plot(epoches, train_loss)
    plt.plot(epoches, val_loss)
    plt.title(title)
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    plt.xlabel('epoch')
    if save_path:
        plt.savefig(save_path)
    if is_show:
        plt.show()

def output_config(config, config_output_path):
    df = pd.DataFrame(list(config.items()))
    df.to_csv(config_output_path, index=False, header=False)