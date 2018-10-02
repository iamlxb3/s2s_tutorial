import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttnDecoderRNN(nn.Module):
    def __init__(self,
                 hidden_size, output_size,
                 n_layers=1, dropout_p=0.1,
                 max_length=None
                 ):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.output_size + 1, self.hidden_size)  # TODO, add to vocab
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = embedded.view(embedded.shape[1], embedded.shape[0], -1)
        embedded = self.dropout(embedded)

        decoder_input_conca = torch.cat((embedded, hidden), 2)
        attn_weights = F.softmax(self.attn(decoder_input_conca), dim=2)  # torch.Size([1, batch_size, decoder_length])

        attn_weights = attn_weights.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)  # torch.Size([batch_size, 1, feature_dim])
        attn_applied = attn_applied.permute(1, 0, 2)
        output = torch.cat((embedded, attn_applied), 2)

        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output), dim=2)

        attn_weights = attn_weights.permute(1, 0, 2)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
