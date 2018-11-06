import pandas as pd
import sys

sys.path.append("..")
from funcs.gen import EnFraDataSet
from torch.utils.data import DataLoader
from funcs.eval_predict import bleu_compute
from funcs.eval_predict import rogue_compute
from funcs.eval_predict import predict_on_test
import numpy as np
import torch
from config import cfg
from config import fra_vocab


def predict():
    # load models
    encoder = torch.load(cfg.encoder_path)
    decoder = torch.load(cfg.decoder_path)
    print("Load encoder from {}, decoder from {}".format(cfg.encoder_path, cfg.decoder_path))
    #

    seq_csv_path = cfg.test_seq_csv_path

    X = pd.read_csv(seq_csv_path)['source'].values
    Y = pd.read_csv(seq_csv_path)['target'].values
    uids = pd.read_csv(seq_csv_path)['uid'].values

    # get generator
    test_generator = EnFraDataSet(X, Y, uids, cfg.encoder_pad_shape, cfg.decoder_pad_shape,
                                  cfg.src_pad_token, cfg.target_pad_token, cfg.use_pretrain_embedding)
    test_loader = DataLoader(test_generator,
                             batch_size=1,
                             shuffle=False,
                             num_workers=cfg.num_workers,
                             )
    test_loss = []
    rogues = []
    bleus = []

    for i, (src_tensor, target_tensor, uid) in enumerate(test_loader):
        loss, decoded_words, target_words = predict_on_test(cfg, encoder, decoder, src_tensor, target_tensor, fra_vocab)

        print("-----------------------------------------------------")
        print("loss: ", loss)
        print("target_words: ", target_words)
        print("Decoded_words: ", decoded_words)

        # TODO, add language model
        # target_words = eval(y_df[(y_df.id == val_id)]['index'].values[0])
        #

        # compute rogue & bleu
        rogue = rogue_compute(target_words, decoded_words)
        bleu = bleu_compute(target_words, decoded_words)
        #
        #
        test_loss.append(loss)
        rogues.append(rogue)
        bleus.append(bleu)

    print("test_loss: ", np.average(test_loss))
    print("rogues: ", np.average(rogues))
    print("bleus: ", np.average(bleus))


if __name__ == '__main__':
    predict()
