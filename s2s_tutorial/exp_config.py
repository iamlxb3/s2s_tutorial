from s2s_config import cfg
import copy
import os

# ------------------------------------------------------
# set experiment 1
# ------------------------------------------------------
exp1 = copy.copy(cfg)
exp1.name = 'exp1'
exp1.data_set = 's2s_toy_data_reverse'
exp1.exp_dir = os.path.join(exp1.results_dir, exp1.name)
# ------------------------------------------------------


experiments = {'exp1': exp1}
