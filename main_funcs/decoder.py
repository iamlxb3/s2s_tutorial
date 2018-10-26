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

        :param yt: torch.Size([1, batch_size, feature_dim])
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
