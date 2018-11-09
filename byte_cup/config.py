import pickle
import torch
import os
import torch.nn as nn

from easydict import EasyDict as edict
from utils.helpers import seq_max_length_get

# config dict
cfg = edict()

# other config
cfg.verbose = True
cfg.load_model = False
cfg.model_type = 'basic_attn' # basic_rnn, basic_attn
cfg.use_pretrain_embedding = False
cfg.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
#

# data-set config
data_set = 'bytecup2018'
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(top_dir, 'data')
model_pkl_dir = os.path.join(top_dir, 'model_pkls')
train_x_dir = os.path.join(data_dir, data_set, 'train')
cfg.train_seq_csv_path = os.path.join(data_dir, data_set, 'train_small.csv')
cfg.test_seq_csv_path = os.path.join(data_dir, data_set, 'test_small.csv')
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
cfg.vocab_size = len(vocab)
#

# data loader config
cfg.num_workers = 8  # num workers for data-loader
cfg.data_shuffle = False
#

# model path config
cfg.encoder_path = os.path.join(model_pkl_dir, '{}_encoder.pkl'.format(data_set))  # set encoder path
cfg.decoder_path = os.path.join(model_pkl_dir, '{}_decoder.pkl'.format(data_set))  # set decoder path
#

# model hyper-parameters
dim = 256
cfg.rnn_type = 'gru' # rnn
cfg.encoder_input_dim = dim
cfg.encoder_hidden_dim = dim
cfg.decoder_input_dim = dim
cfg.decoder_hidden_dim = dim
cfg.encoder_pad_shape = (seq_max_length_get(cfg.train_seq_csv_path, 'source'), 1)
cfg.decoder_pad_shape = (seq_max_length_get(cfg.train_seq_csv_path, 'target'), 1)
cfg.share_embedding = True # encoder and decoder share the same embedding layer
cfg.softmax_share_embedd = True
cfg.encoder_bi_direction = True
cfg.is_coverage = False
cfg.coverage_loss_coeff = 0.2
cfg.is_point_generator = True

if cfg.is_coverage:
    cfg.attn_method = 'coverage'
else:
    cfg.attn_method = 'general'
#

# training hyper-parameters config
cfg.lr = 1e-3
cfg.epoches = 1
cfg.batch_size = 2
cfg.use_teacher_forcing = True
cfg.teacher_forcing_ratio = 0.3
cfg.criterion = nn.NLLLoss(ignore_index=cfg.target_pad_token)
#
