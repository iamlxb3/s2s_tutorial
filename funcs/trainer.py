"""
训练的两个主要的函数都封装在这里了
"""

import ipdb
import sys
import torch
import random
import numpy as np

sys.path.append('..')
from utils.helpers import save_cktpoint
from utils.helpers import encode_func
from utils.helpers import decode_func
from funcs.eval_predict import eval_on_val


def train_1_batch(cfg, input_tensor, target_tensor, encoder, decoder):
    """
    训练一个batch的数据
    """
    # load config，读入一些配置
    optimizer = cfg.optimizer
    teacher_forcing_ratio = cfg.teacher_forcing_ratio
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #

    # initialize，偏导数归零
    loss = 0
    optimizer.zero_grad()
    #

    # encode，对输入做encode
    # encoder_outputs: 序列长度 x batch_size x hidden_dim
    encoder_outputs, encoder_last_hidden = encode_func(cfg, input_tensor, encoder)
    #

    # decode
    loss, target_max_len = decode_func(cfg, loss, target_tensor, encoder_outputs, encoder_last_hidden,
                                       use_teacher_forcing, decoder, input_tensor=input_tensor)
    #

    # calculate gradient & update parameters，更新参数
    loss.backward()
    optimizer.step()
    #

    # this is the reduced form of loss
    return loss


def epoches_train(cfg, train_loader, val_loader, encoder, decoder, epoch_recorder, encoder_path, decoder_path):
    """
    所有的训练在这里完成，跑完所有的epoch
    """
    # add attention-recoder

    for epoch, epoch_index in enumerate(range(cfg.epoches)):

        # set to train mode，将模型设定为train模式
        encoder, decoder = encoder.train(), decoder.train()

        epoch_loss = 0
        for batch_index, (batch_x, batch_y, uid) in enumerate(train_loader):

            # 读取一个batch的数据
            batch_x = batch_x.to(cfg.device)
            batch_y = batch_y.to(cfg.device)

            # 训练一个batch的数据
            loss = train_1_batch(cfg, batch_x, batch_y, encoder, decoder)

            # 累计loss
            epoch_loss += loss

            if cfg.verbose:
                print("Epoch-{} batch_index-{}/{} Loss: {}".format(epoch, batch_index, len(train_loader), loss))

        # eval on validation set，每一个epoch训练好都要在验证集上测试一下
        # set to eval mode
        encoder, decoder = encoder.eval(), decoder.eval()
        val_loss = []
        for X_val, Y_val, uid in val_loader:
            # evaluate on validation set, use_teacher_forcing only for debug
            batch_loss = eval_on_val(cfg, encoder, decoder, X_val, Y_val, use_teacher_forcing=False)
            val_loss.append(batch_loss)
        val_loss = np.average(val_loss)
        #

        # lr_scheduler，根据验证集上的表现，决定要不要调整模型的学习率
        cfg.lr_scheduler.step(val_loss)
        print("Current lr: ", cfg.optimizer.param_groups[0]['lr'])
        #

        # epoch print
        epoch_loss = epoch_loss / cfg.step_size
        print("Epoch: {}, loss: {}, val_loss: {}".format(epoch, epoch_loss, val_loss))
        #

        # Save checkpoint，根据验证集上的表现，来保存当前最好的模型
        lowest_val_loss, lowest_val_loss_index = epoch_recorder.lowest_val_loss
        if val_loss < lowest_val_loss:
            save_cktpoint(encoder, decoder, encoder_path, decoder_path)
            print("val_loss: {}, epoch-{}, Save checkpoint to {}, {}".format(val_loss, epoch_index, encoder_path,
                                                                             decoder_path))
        else:
            print("val_loss no improvement, lowest: {}, epoch-{}".format(lowest_val_loss, lowest_val_loss_index))
        epoch_recorder.val_loss_update(val_loss)
        epoch_recorder.train_loss_update(epoch_loss)
        #
