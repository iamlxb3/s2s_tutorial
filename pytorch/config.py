import torch

class global_var:
    '''需要定义全局变量的放在这里，最好定义一个初始值'''
    MAX_LENGTH = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SOS_TOKEN = 0
    EOS_TOKEN = 1
    teacher_forcing_ratio = 0.5

# MAX_LENGTH get_value & set_value
def set_max_len(MAX_LENGTH):
    global_var.MAX_LENGTH = MAX_LENGTH
def get_max_len():
    return global_var.MAX_LENGTH

# device get_value & set_value
def set_device(device):
    global_var.device = device
def get_device():
    return global_var.device

# SOS_TOKEN get_value & set_value
def set_sos_token(SOS_TOKEN):
    global_var.SOS_TOKEN = SOS_TOKEN
def get_sos_token():
    return global_var.SOS_TOKEN

# EOS_TOKEN get_value & set_value
def set_eos_token(EOS_TOKEN):
    global_var.EOS_TOKEN = EOS_TOKEN
def get_eos_token():
    return global_var.EOS_TOKEN

# teacher_forcing_ratio get_value & set_value
def set_teacher_forcing_ratio(teacher_forcing_ratio):
    global_var.teacher_forcing_ratio = teacher_forcing_ratio
def get_teacher_forcing_ratio():
    return global_var.teacher_forcing_ratio
