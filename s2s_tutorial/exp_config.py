import copy
import os
import sys

sys.path.append('..')

from s2s_config import cfg as default_cfg
from utils.helpers import auto_config_path_etc

# ------------------------------------------------------
# set experiment 1 - for eng_fra dataset
# ------------------------------------------------------
exp1 = copy.copy(default_cfg)

# config dict
exp1.name = 'eng_fra'

# other config
exp1.verbose = True

# data-set config
exp1.data_set = 'eng_fra'
exp1.train_csv_name = 'train_small_seq.csv'
exp1.test_csv_name = 'test_small_seq.csv'
exp1.seq_min_len = 1  # filter the src samples longer than max_len
exp1.seq_max_len = 50

# vocab config
exp1.src_vocab_name = 'small_eng_vocab.pkl'
exp1.target_vocab_name = 'small_fra_vocab.pkl'

# model path config
exp1.encoder_path = ('../model_pkls/{}_encoder.pkl'.format(exp1.data_set))  # set encoder path
exp1.decoder_path = ('../model_pkls/{}_decoder.pkl'.format(exp1.data_set))  # set decoder path

# model hyper-parameters
exp1.share_embedding = False
exp1.encode_rnn_type = 'rnn'  # rnn, gru
exp1.decode_rnn_type = 'basic_attn'  # basic_rnn, basic_attn
exp1.share_embedding = False

# training hyper-parameters config
exp1.epoches = 5
exp1.batch_size = 64
exp1.use_teacher_forcing = True
exp1.teacher_forcing_ratio = 0.3

exp1 = auto_config_path_etc(exp1)
# ------------------------------------------------------


experiments = {'exp1': exp1}
