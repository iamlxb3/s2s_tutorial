"""
项目配置文件，绝大部分的内容都在这里配置，除了优化器和修改LR的scheduler，在run_train_val.py中配置。
"""

import pickle
import torch
import os
import torch.nn as nn

from easydict import EasyDict as edict
from utils.helpers import seq_max_length_get

# config dict
cfg = edict()

# other config 一些通用的配置
cfg.verbose = True
cfg.load_model = False
cfg.model_type = 'basic_attn'  # 别改
cfg.use_pretrain_embedding = False
cfg.device = torch.device("cpu")  # TODO, 如果用gpu把这里改成gpu
cfg.plot_attn = False # 在测试的时候是否plot attention
#

# data-set config 数据集配置，这里没啥好改的
data_set = 'bytecup2018'
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(top_dir, 'data')
model_pkl_dir = os.path.join(top_dir, 'model_pkls')
train_x_dir = os.path.join(data_dir, data_set, 'train')
cfg.train_seq_csv_path = os.path.join(data_dir, data_set, 'train_small.csv')
cfg.test_seq_csv_path = os.path.join(data_dir, data_set, 'test_small.csv')
vocab_path = os.path.join(data_dir, data_set, 'vocab.pkl')
#

# vocab config，这里也没啥好改的
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

# data loader config，这里也也没啥好改的
cfg.num_workers = 8  # num workers for data-loader
cfg.data_shuffle = True # 是否每一个epoch打乱数据
#

# model path config
cfg.encoder_path = os.path.join(model_pkl_dir, '{}_encoder.pkl'.format(data_set))  # set encoder path
cfg.decoder_path = os.path.join(model_pkl_dir, '{}_decoder.pkl'.format(data_set))  # set decoder path
#

# model hyper-parameters
dim = 256 # TODO, 设置dim，我图方便就放一块了，你可以把他们都分开
cfg.rnn_type = 'gru'  # rnn
cfg.encoder_input_dim = dim
cfg.encoder_hidden_dim = dim
cfg.decoder_input_dim = dim
cfg.decoder_hidden_dim = dim
cfg.encoder_pad_shape = (seq_max_length_get(cfg.train_seq_csv_path, 'source'), 1)
cfg.decoder_pad_shape = (seq_max_length_get(cfg.train_seq_csv_path, 'target'), 1)
cfg.share_embedding = True  # encoder and decoder share the same embedding layer，encoder,decoder的输入是否共享embedding层
cfg.softmax_share_embedd = False # encoder softmax的输出是否共享一部分的embedding参数，这里有一个投影矩阵
cfg.encoder_bi_direction = False # encoder是否双向
cfg.is_coverage = False # 是否使用coverage功能
cfg.coverage_loss_coeff = 0.2 # coverage loss占比
cfg.is_point_generator = False # 是否使用pointer_generator

# 这里不用改
if cfg.is_coverage:
    cfg.attn_method = 'coverage'
else:
    cfg.attn_method = 'general'
#

# training hyper-parameters config, TODO, 这里是可以调的参数
cfg.lr = 1e-3 # 学习率
cfg.epoches = 1 # 学多少轮
cfg.batch_size = 2
cfg.use_teacher_forcing = True
cfg.teacher_forcing_ratio = 0.3 # 百分之多少的几率使用teacher-forcing
cfg.criterion = nn.NLLLoss(ignore_index=cfg.target_pad_token) # 损失函数，暂时可以不动
cfg.decode_mode = 'greedy' # beam_search, greedy, decoder 解码的方式，beam_search 太慢了，先用greedy吧
cfg.beam_width = 1 # 这个先不用改
#
