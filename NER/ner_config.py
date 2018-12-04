"""
TODO: 1. make easy_dict frozen:https://stackoverflow.com/questions/25247981/python-freeze-dict-keys-after-creation
"""
import torch
import torch.nn as nn

from easydict import EasyDict as edict
from utils.helpers import auto_config_path_etc

# config dict
cfg = edict()
cfg.name = 'ner'  # default name for config

# other config
cfg.randseed = 1
cfg.verbose = True
cfg.load_model = False
cfg.use_pretrain_embedding = False
cfg.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.plot_attn = False
cfg.plot_loss = False
#

# path config
cfg.results_dir = '../results'
cfg.data_dir = '../data'
#

# data-set config
cfg.data_set = 'CoNLL_2003'
cfg.train_csv_name = 'train.csv'
cfg.test_csv_name = 'test.csv'
cfg.seq_min_len = 1  # filter the src samples longer than max_len
cfg.seq_max_len = 5  # filter the src samples longer than max_len
cfg.val_percent = 0.2
cfg.is_index_input = False # whether the input is represented by index or in high-dimension
cfg.input_max_len = 50 if not cfg.is_index_input else None # TODO, config
cfg.train_npy_dir_name = 'train'  # the abs path for train_pt_dir will be updated
cfg.test_npy_dir_name = 'test'  # the abs path for train_pt_dir will be updated
#

# vocab config
cfg.src_vocab_name = 'dummy_src_vocab.pkl'
cfg.target_vocab_name = 'NER_vocab.pkl'
#

# data loader config
cfg.num_workers = 8  # num workers for data-loader
cfg.data_shuffle = True
#

# model path config
cfg.encoder_path = ('../model_pkls/{}_encoder.pkl'.format(cfg.data_set))  # set encoder path
#

# model hyper-parameters
cfg.encode_rnn_type = 'rnn'  # rnn, gru
cfg.encoder_input_dim = 74 # TODO, config
cfg.encoder_hidden_dim = 128
cfg.softmax_share_embedd = False
cfg.encoder_bi_direction = True


# training hyper-parameters config
cfg.lr = 1e-3
cfg.epoches = 2
cfg.batch_size = 512
cfg.test_batch_size = 1
cfg.use_teacher_forcing = False
cfg.teacher_forcing_ratio = 0.0
cfg.decode_mode = 'greedy'  # beam_search, greedy
cfg.beam_width = 1  #
#
cfg.criterion_cls = nn.NLLLoss
cfg = auto_config_path_etc(cfg)
