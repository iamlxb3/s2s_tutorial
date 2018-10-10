import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb



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
        self.decoder_dim = self.hidden_size # TODO, fix
        self.embedding = nn.Embedding(self.output_size + 1, self.hidden_size)  # TODO, add to vocab
        self.attn = nn.Linear(self.hidden_size + self.decoder_dim, 1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """

        :param input: torch.Size([batch_size, 1])
        :param hidden: torch.Size([1, batch_size, hidden_dim])
        :param encoder_outputs: torch.Size([encoder_length, batch_size, 256])
        :return:
        """

        # TODO, clean up code
        
        decoder_y_input = self.embedding(input)
        decoder_y_input = torch.transpose(decoder_y_input, 0, 1)
        # decoder_y_input = self.dropout(decoder_y_input) # torch.Size([1, batch_size, feature_dim])

        attn_weights = []
        #print("---------------------")
        for pos, encoder_output_t in enumerate(encoder_outputs):
            # encoder_output_t : torch.Size([batch_size, feature_dim])
            encoder_o_t_decoder_i = torch.cat((hidden,  encoder_output_t.view(1, encoder_output_t.size(0), -1)), 2)
            #attn_weight = F.softmax(self.attn(encoder_o_t_decoder_i), dim=2)  # torch.Size([1, batch_size, 1])
            attn_weight = self.attn(encoder_o_t_decoder_i) # torch.Size([1, batch_size, 1])
            attn_weights.append(attn_weight)

            # if pos == 0:
            #     print("pos-{}, attn_weight-{}".format(pos, attn_weight))
            # #     print("hidden: {}, encoder_output_t: {}, attn_weight: {}"
            #           .format(hidden[:,:,:10],encoder_output_t[:,:10], float(attn_weight)))
            #print("encoder_o_t_decoder_i: {}, attn_weight: {}".format(torch.sum(encoder_o_t_decoder_i), torch.sum(attn_weight)))

        attn_weights = torch.cat(attn_weights, 0)
        attn_weights = F.softmax(attn_weights, 0) # TODO

        # >> > batch1 = torch.randn(10, 3, 4)
        # >> > batch2 = torch.randn(10, 4, 5)
        # >> > res = torch.bmm(batch1, batch2)
        # >> > res.size()
        # torch.Size([10, 3, 5])

        #ipdb.set_trace()

        context_vector = torch.bmm(attn_weights.permute(1, 2, 0), encoder_outputs.permute(1, 0, 2))  # torch.Size([batch_size, 1, feature_dim])
        context_vector = context_vector.permute(1, 0, 2)

        # TODO, temp
        output = torch.cat((decoder_y_input, context_vector), 2)
        output = F.relu(self.attn_combine(output))
        #

        #output = context_vector #TODO

        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output), dim=2)

        attn_weights = attn_weights.permute(1, 0, 2)
        return output, hidden, attn_weights, context_vector

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

