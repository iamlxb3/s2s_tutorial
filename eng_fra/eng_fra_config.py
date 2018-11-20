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
cfg.use_pretrain_embedding = False
cfg.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.plot_attn = True
#

# data-set config
data_set = 'eng_fra'
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(top_dir, 'data')
model_pkl_dir = os.path.join(top_dir, 'model_pkls')
cfg.train_seq_csv_path = os.path.join(data_dir, data_set, 'train_small_seq.csv')
cfg.test_seq_csv_path = os.path.join(data_dir, data_set, 'test_small_seq.csv')
en_vocab_path = os.path.join(data_dir, data_set, 'small_eng_vocab.pkl')
fra_vocab_path = os.path.join(data_dir, data_set, 'small_fra_vocab.pkl')
#

# vocab config
en_vocab = pickle.load(open(en_vocab_path, 'rb'))
fra_vocab = pickle.load(open(fra_vocab_path, 'rb'))
cfg.src_vocab_len = len(en_vocab)
cfg.target_vocab_len = len(fra_vocab)
cfg.src_pad_token = int(en_vocab.index('<PAD>'))
cfg.target_SOS_token = int(fra_vocab.index('<SOS>'))
cfg.target_EOS_token = int(fra_vocab.index('<SOS>'))
cfg.target_pad_token = int(fra_vocab.index('<SOS>'))
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
dim = 512
cfg.decode_rnn_type = 'basic_attn' # basic_rnn, basic_attn # decode_rnn_type
cfg.encode_rnn_type = 'gru' # rnn, gru
cfg.encoder_input_dim = dim
cfg.encoder_hidden_dim = dim
cfg.decoder_input_dim = dim
cfg.decoder_hidden_dim = dim
cfg.encoder_pad_shape = (seq_max_length_get(cfg.train_seq_csv_path, 'source'), 1)
cfg.decoder_pad_shape = (seq_max_length_get(cfg.train_seq_csv_path, 'target'), 1)
cfg.softmax_share_embedd = False
cfg.share_embedding = False  # encoder and decoder share the same embedding layer
cfg.encoder_bi_direction = False
cfg.is_coverage = False
cfg.coverage_loss_coeff = 0.0
cfg.is_point_generator = False

if cfg.is_coverage:
    cfg.attn_method = 'coverage'
else:
    cfg.attn_method = 'general'
#
#

# training hyper-parameters config
cfg.lr = 1e-3
cfg.epoches = 1
cfg.batch_size = 16
cfg.test_batch_size = 1
cfg.use_teacher_forcing = True
cfg.teacher_forcing_ratio = 0.1
cfg.criterion = nn.NLLLoss(ignore_index=cfg.target_pad_token)
cfg.decode_mode = 'greedy' # beam_search, greedy
cfg.beam_width = 1 #

#
