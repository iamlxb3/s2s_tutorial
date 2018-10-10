import torch
import ipdb
import torch.nn as nn



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_size, self.n_layers)
        self.x_max_length = 150

    def forward(self, input, h0):
        output, hidden = self.gru(input, h0)
        return output, hidden

    def initHidden(self, batch, device):
        return torch.zeros(1, batch, self.hidden_size, device=device)
