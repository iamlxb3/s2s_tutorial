from s2s_config import cfg
import copy
import os

def refresh_configs(cfg):

    # TODO ,add more

    # model path config
    cfg.encoder_path = ('../model_pkls/{}_encoder.pkl'.format(cfg.data_set))  # set encoder path
    cfg.decoder_path = ('../model_pkls/{}_decoder.pkl'.format(cfg.data_set))  # set decoder path
    #

    cfg.exp_dir = os.path.join(cfg.results_dir, cfg.name)



# ------------------------------------------------------
# set experiment 1
# ------------------------------------------------------
exp1 = copy.copy(cfg)
exp1.name = 'eng_fra'
exp1.data_set = 'eng_fra'
exp1.exp_dir = os.path.join(exp1.results_dir, exp1.name)
exp1.seq_max_len = 50
exp1.share_embedding = False
exp1.encode_rnn_type = 'attn'
exp1.epoches = 20
exp1.use_teacher_forcing = True
exp1.teacher_forcing_ratio = 0.5
# ------------------------------------------------------


experiments = {'exp1': exp1}
