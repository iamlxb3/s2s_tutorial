import torch
import ipdb
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_size, self.n_layers)

    def forward(self, input, hidden):
        input = input.view(1, 1, -1)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
