from s2s_config import cfg
import copy
import os
import pickle

def refresh_configs(exp):

    # TODO ,add more

    # model path config
    exp.encoder_path = ('../model_pkls/{}_encoder.pkl'.format(exp.data_set))  # set encoder path
    exp.decoder_path = ('../model_pkls/{}_decoder.pkl'.format(exp.data_set))  # set decoder path
    #

    exp.exp_dir = os.path.join(exp.results_dir, exp.name)

    src_vocab = pickle.load(open(exp1.src_vocab_path, 'rb'))
    target_vocab = pickle.load(open(exp1.target_vocab_path, 'rb'))
    exp.src_vocab_len = len(src_vocab)
    exp.target_vocab_len = len(target_vocab)
    exp.src_pad_token = int(src_vocab.index('<PAD>'))
    exp.target_SOS_token = int(target_vocab.index('<SOS>'))
    exp.target_EOS_token = int(target_vocab.index('<SOS>'))
    exp.target_pad_token = int(target_vocab.index('<SOS>'))

    return exp

# ------------------------------------------------------
# set experiment 1 - for eng_fra dataset
# ------------------------------------------------------
exp1 = copy.copy(cfg)
exp1.name = 'eng_fra'
exp1.data_set = 'eng_fra'
exp1.exp_dir = os.path.join(exp1.results_dir, exp1.name)
exp1.seq_min_len = 1  # filter the src samples longer than max_len
exp1.seq_max_len = 50
exp1.share_embedding = False
exp1.encode_rnn_type = 'rnn' # rnn, gru
exp1.decode_rnn_type = 'basic_attn'  # basic_rnn, basic_attn
exp1.epoches = 5
exp1.batch_size = 64
exp1.use_teacher_forcing = True
exp1.teacher_forcing_ratio = 0.3
exp1.verbose = True
exp1.share_embedding = False

exp1.train_seq_csv_path = os.path.join(exp1.data_dir, exp1.data_set, 'train_small_seq.csv')
exp1.test_seq_csv_path = os.path.join(exp1.data_dir, exp1.data_set, 'test_small_seq.csv')
exp1.src_vocab_path = os.path.join(exp1.data_dir, exp1.data_set, 'small_eng_vocab.pkl')
exp1.target_vocab_path = os.path.join(exp1.data_dir, exp1.data_set, 'small_fra_vocab.pkl')

exp1 = refresh_configs(exp1)
# ------------------------------------------------------


experiments = {'exp1': exp1}
