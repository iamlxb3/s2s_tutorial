import ipdb
import copy
import os
import sys
import inspect

sys.path.append('..')

from ner_config import cfg as default_cfg
from utils.helpers import auto_config_path_etc


# # ------------------------------------------------------
# # set experiment 1 - for eng_fra dataset
# # ------------------------------------------------------
# exp1 = copy.copy(default_cfg)
#
# # config dict
# exp1.name = 'eng_fra'
#
# # other config
# exp1.verbose = True
#
# # data-set config
# exp1.data_set = 'eng_fra'
# exp1.train_csv_name = 'train_small_seq.csv'
# exp1.test_csv_name = 'test_small_seq.csv'
# exp1.seq_min_len = 1  # filter the src samples longer than max_len
# exp1.seq_max_len = 50
#
# # vocab config
# exp1.src_vocab_name = 'small_eng_vocab.pkl'
# exp1.target_vocab_name = 'small_fra_vocab.pkl'
#
# # model path config
# exp1.encoder_path = ('../model_pkls/{}_encoder.pkl'.format(exp1.data_set))  # set encoder path
# exp1.decoder_path = ('../model_pkls/{}_decoder.pkl'.format(exp1.data_set))  # set decoder path
#
# # model hyper-parameters
# exp1.encode_rnn_type = 'rnn'  # rnn, gru
# exp1.decode_rnn_type = 'basic_attn'  # basic_rnn, basic_attn
# exp1.share_embedding = False
# exp1.encoder_input_dim = 128
# exp1.decoder_input_dim = 128
# exp1.encoder_hidden_dim = 256
# exp1.decoder_hidden_dim = 256
#
# # training hyper-parameters config
# exp1.epoches = 20
# exp1.batch_size = 32
# exp1.use_teacher_forcing = True
# exp1.teacher_forcing_ratio = 0.3
#
# exp1 = auto_config_path_etc(exp1)
# # TODO, add lock after
# # ------------------------------------------------------
#


# ------------------------------------------------------
# set experiment 2 - multiple sequence -> 1 output
# ------------------------------------------------------
def exp2():
    cfg = copy.copy(default_cfg)
    cfg.name = 'exp2'
    cfg.data_set = 's2s_toy_data_copy'
    cfg.train_csv_name = 'train.csv'
    cfg.test_csv_name = 'test.csv'
    cfg.is_index_input = False
    cfg.batch_size = 512
    cfg.verbose = True
    cfg.is_index_input = False  # whether the input is represented by index or in high-dimension
    cfg.input_max_len = 50 if not cfg.is_index_input else None
    cfg.encoder_input_dim = 30
    cfg = auto_config_path_etc(cfg)
    return cfg


# ------------------------------------------------------
#
class Experiments:
    def __init__(self):
        self._experiments = {}
        self._add_all_experiments()

    @property
    def experiments(self):
        return self._experiments

    @property
    def keys(self):
        return self._experiments.keys()

    def __call__(self, exp_name):
        return self._experiments[exp_name]()

    def _add_all_experiments(self):
        funcs = inspect.getmembers(sys.modules[__name__],
                                   predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__)
        for func_tuple in funcs:
            func_name, func = func_tuple
            self._experiments[func_name] = func


experiments = Experiments()
