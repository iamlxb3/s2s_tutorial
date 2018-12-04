import ipdb
import sys
import torch
import random
import numpy as np

sys.path.append('..')
from utils.helpers import save_cktpoint
from utils.helpers import encode_func
from utils.helpers import loss_compute


def train_val_1_batch(cfg, input_tensor, target_tensor, encoder, is_val=False):
    # load config
    optimizer = cfg.optimizer
    #

    # initialize
    loss = 0
    if not is_val:
        optimizer.zero_grad()
    #

    # encode
    Y_predict, encoder_last_hidden = encode_func(cfg, input_tensor, encoder)
    #

    # compute loss
    loss = loss_compute(loss, cfg, Y_predict, target_tensor)
    #

    # calculate gradient & update parameters
    if not is_val:
        loss.backward()
        optimizer.step()
    #

    # this is the reduced form of loss
    return float(loss)


def epoches_train(cfg, train_loader, val_loader, encoder, epoch_recorder, encoder_path):
    # add attention-recoder

    for epoch, epoch_index in enumerate(range(cfg.epoches)):

        # set to train mode
        encoder = encoder.train()

        epoch_loss = 0
        for batch_index, (batch_x, batch_y, uid) in enumerate(train_loader):
            batch_x = batch_x.to(cfg.device)
            batch_y = batch_y.to(cfg.device)

            # ipdb > batch_x.shape
            # torch.Size([32, 50, 100]), torch.Size([32, 50, 1])
            # ipdb > batch_y.shape
            # torch.Size([32, 4, 1])

            loss = train_val_1_batch(cfg, batch_x, batch_y, encoder)

            epoch_loss += loss

            if cfg.verbose:
                print("Epoch-{} batch_index-{}/{} Loss: {}".format(epoch, batch_index, len(train_loader), loss))

        # set to eval mode
        encoder = encoder.eval()
        val_loss = []
        for X_val, Y_val, uid in val_loader:
            with torch.no_grad():
                loss = train_val_1_batch(cfg, X_val, Y_val, encoder, is_val=True)
            val_loss.append(loss)
        val_loss = np.average(val_loss)
        #

        # lr_scheduler
        cfg.lr_scheduler.step(val_loss)
        print("Current lr: ", cfg.optimizer.param_groups[0]['lr'])
        #

        # epoch print
        epoch_loss = epoch_loss / cfg.step_size
        print("Epoch: {}, loss: {}, val_loss: {}".format(epoch, epoch_loss, val_loss))
        #

        # Save checkpoint
        lowest_val_loss, lowest_val_loss_index = epoch_recorder.lowest_val_loss
        if val_loss < lowest_val_loss:
            save_cktpoint(encoder, encoder_path)
            print("val_loss: {}, epoch-{}, Save checkpoint to {}".format(val_loss, epoch_index, encoder_path))
        else:
            print("val_loss no improvement, lowest: {}, epoch-{}".format(lowest_val_loss, lowest_val_loss_index))
        epoch_recorder.val_loss_update(val_loss)
        epoch_recorder.train_loss_update(epoch_loss)
        #
