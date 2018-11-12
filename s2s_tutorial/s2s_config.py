import pickle
import torch
import os
import torch.nn as nn

from easydict import EasyDict as edict
from utils.helpers import seq_max_length_get

# config dict
cfg = edict()

# other config
cfg.verbose = False
cfg.load_model = False
cfg.model_type = 'basic_rnn'  # basic_rnn, basic_attn
cfg.use_pretrain_embedding = False
cfg.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.plot_attn = False
#

# data-set config
data_set = 's2s_toy_data_copy'
cfg.seq_max_len = 5  # filter the src samples longer than max_len
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(top_dir, 'data')
model_pkl_dir = os.path.join(top_dir, 'model_pkls')
train_x_dir = os.path.join(data_dir, data_set, 'train')
cfg.train_seq_csv_path = os.path.join(data_dir, data_set, 'train.csv')
cfg.test_seq_csv_path = os.path.join(data_dir, data_set, 'test.csv')
vocab_path = os.path.join(data_dir, data_set, 'vocab.pkl')
#

# vocab config
vocab = pickle.load(open(vocab_path, 'rb'))
src_vocab_len = len(vocab)
src_pad_token = int(vocab.index('<PAD>'))
target_vocab_len = len(vocab)
target_SOS_token = int(vocab.index('<SOS>'))
target_EOS_token = int(vocab.index('<EOS>'))
target_pad_token = int(vocab.index('<PAD>'))
cfg.src_vocab_len = src_vocab_len
cfg.target_vocab_len = target_vocab_len
cfg.src_pad_token = src_pad_token
cfg.target_SOS_token = target_SOS_token
cfg.target_EOS_token = target_EOS_token
cfg.target_pad_token = target_pad_token
#

# data loader config
cfg.num_workers = 8  # num workers for data-loader
cfg.data_shuffle = True
#

# model path config
cfg.encoder_path = os.path.join(model_pkl_dir, '{}_encoder.pkl'.format(data_set))  # set encoder path
cfg.decoder_path = os.path.join(model_pkl_dir, '{}_decoder.pkl'.format(data_set))  # set decoder path
#

# model hyper-parameters
cfg.rnn_type = 'rnn'
cfg.encoder_input_dim = 32
cfg.encoder_hidden_dim = 256
cfg.decoder_hidden_dim = 256
cfg.encoder_pad_shape = (seq_max_length_get(cfg.train_seq_csv_path, 'source'), 1)
cfg.decoder_pad_shape = (seq_max_length_get(cfg.train_seq_csv_path, 'target'), 1)
cfg.softmax_share_embedd = False
cfg.share_embedding = True  # encoder and decoder share the same embedding layer
if cfg.share_embedding:
    cfg.decoder_input_dim = cfg.encoder_input_dim
else:
    cfg.decoder_input_dim = 256
cfg.encoder_bi_direction = False
cfg.is_coverage = False
cfg.coverage_loss_coeff = 0.0

if cfg.is_coverage:
    cfg.attn_method = 'coverage'
else:
    cfg.attn_method = 'general'
#
#

# training hyper-parameters config
cfg.lr = 1e-3
cfg.epoches = 20
cfg.batch_size = 32
cfg.test_batch_size = 1
cfg.use_teacher_forcing = False
cfg.teacher_forcing_ratio = 0.0
cfg.criterion = nn.NLLLoss(ignore_index=cfg.target_pad_token)
cfg.decode_mode = 'greedy'  # beam_search, greedy
cfg.beam_width = 1  #
#
