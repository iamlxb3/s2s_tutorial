from easydict import EasyDict as edict
from utils.helpers import seq_max_length_get
import torch
import os


# set path
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(top_dir, 'data')
data_set = 'eng_fra'
seq_csv_path = os.path.join(data_dir, data_set, 'train_small_seq.csv')


# config dict
config = edict()
config.device = torch.device("cpu")
dim = 256
config.encoder_input_dim = dim
config.encoder_hidden_dim = dim
config.decoder_input_dim = dim
config.decoder_hidden_dim = dim
config.encoder_pad_shape = (seq_max_length_get(seq_csv_path, 'source'), 1)
config.decoder_pad_shape = (seq_max_length_get(seq_csv_path, 'target'), 1)
config.lr = 1e-4
config.epoches = 10
config.batch_size = 4
config.num_workers = 8
config.use_teacher_forcing = True
config.data_shuffle = True
config.verbose = True
#
