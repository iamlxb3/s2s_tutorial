import numpy as np
import keras
import math
import random
import ipdb
from seq2seq.models import SimpleSeq2Seq


def random_gen(X, Y, batch_size=10):
    while True:
        random.shuffle(X)
        random.shuffle(Y)
        x_batch = []
        y_batch = []
        for i, x in enumerate(X):
            y = Y[i]
            y = keras.utils.to_categorical(y, num_classes=8)
            y = np.array([y])
            x = np.array([x])
            x_batch.append(x)
            y_batch.append(y)

            assert len(x_batch) == len(y_batch)

            if len(x_batch) == batch_size:
                x_batch = np.concatenate(x_batch)
                y_batch = np.concatenate(y_batch)
                yield (x_batch, y_batch)
                x_batch = []
                y_batch = []


def model():
    model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8, depth=5)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def main():
    # config
    batch_size = 5

    # load model
    model1 = model()

    # load data
    train_N = 20
    train_X = np.random.random((train_N, 10, 5))
    #train_Y = np.random.randint(8, size=(train_N, 8, 1))
    train_Y = np.ones((train_N, 8, 1))

    train_gen = random_gen(train_X, train_Y, batch_size=10)
    train_step_size = int(math.ceil(train_N / batch_size))
    #ipdb.set_trace()
    # train
    model1.fit_generator(generator=train_gen,
                         epochs=100,
                         steps_per_epoch=train_step_size,
                         )


main()