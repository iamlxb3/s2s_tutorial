import torch
import ipdb
import torch.nn as nn
import torch.nn.functional as F


# RNN
class DecoderRnn(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(DecoderRnn, self).__init__()
        n_layers = 1
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, yt, ht):
        """
        :param yt: torch.Size([batch_size, feature_dim])
        :param ht: torch.Size([1, batch_size, feature_dim])
        :return:
        output: torch.Size([1, batch_size, vocab_size])
        hidden: torch.Size([1, batch_size, feature_dim])
        """
        yt = self.embedding(yt)
        yt = torch.transpose(yt, 0, 1)
        output, hidden = self.rnn(yt, ht)
        output = F.log_softmax(self.out(output), dim=2)
        return output, hidden


# Decoder Attention
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat', 'coverage']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))
        elif self.method == 'coverage':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
            self.coverage_param = torch.nn.Linear(1, 1)

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def coverage_score(self, hidden, encoder_output, coverage):
        energy = self.attn(encoder_output)
        coverage_value = torch.transpose(self.coverage_param(coverage.unsqueeze(2)).squeeze(2), 1, 0)
        return torch.sum(hidden * energy, dim=2) * coverage_value

    def forward(self, hidden, encoder_outputs, coverage=None):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'coverage':
            attn_energies = self.coverage_score(hidden, encoder_outputs, coverage)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# pointer-generator
class PointerGenerator(nn.Module):
    """
    inspired by "Get To The Point: Summarization with Pointer-Generator Networks"
    """

    def __init__(self, conca_dim):
        super(PointerGenerator, self).__init__()
        self.conca = nn.Linear(conca_dim, 1)

    def forward(self, conca_v):
        prob = F.sigmoid(self.conca(conca_v))
        # ADD assertion
        return prob


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, vocab_size, input_dim, hidden_size, n_layers=1, dropout=0.1,
                 softmax_share_embedd=False, is_point_generator=False, pad_token=None):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.pad_token = pad_token

        # Define layers
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.attn = Attn(attn_model, hidden_size)
        self.softmax_share_embedd = softmax_share_embedd
        # TODO, may be better initialize?
        if self.softmax_share_embedd:
            self.embedding_project = torch.nn.init.xavier_uniform_(
                torch.randn((hidden_size, input_dim), requires_grad=True))

        self.is_pg = is_point_generator
        if is_point_generator:
            self.pg = PointerGenerator(hidden_size * 2)

    def forward(self, input_step, last_hidden, encoder_outputs, coverage=None):
        """
        :param input_step:
        :param last_hidden:
        :param encoder_outputs:
        :return:
        output.shape : batch_size, vocab_size
        hidden.shape : layer_num * direction_num, batch_size, hidden_dim
        """
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        embedded = torch.transpose(embedded, 0, 1)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs, coverage=coverage)
        # TODO, add softmax?
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6

        # ADD BY PJS, out & embedding weight sharing
        if self.softmax_share_embedd:
            # shape: vocab_size x hidden dim
            output_weights = self.embedding_project @ torch.transpose(self.embedding.weight.detach(), 1, 0)
            output = concat_output @ output_weights
        else:
            output = self.out(concat_output)

        output = F.log_softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden, attn_weights
