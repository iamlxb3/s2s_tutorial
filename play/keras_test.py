import numpy as np
import keras
import math
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


def random_gen(X, Y, batch_size=10):
    while True:
        random.shuffle(X)
        random.shuffle(Y)
        x_batch = []
        y_batch = []
        for i, x in enumerate(X):
            y = Y[i]
            y = keras.utils.to_categorical(y, num_classes=2)
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
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(10,)))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=2, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.00001),
                  metrics=['accuracy'])


def main():
    # config
    batch_size = 10

    # load model
    model1 = model()

    # load data
    train_N = 100
    val_N = 10
    train_X = np.random.random((train_N, 10))
    train_Y = np.random.randint(2, size=(train_N, 1))
    val_X = np.random.random((val_N, 10))
    val_Y = np.random.randint(2, size=(val_N, 1))
    train_gen = random_gen(train_X, train_Y, batch_size=batch_size)
    train_step_size = int(math.ceil(train_N / batch_size))
    val_gen = random_gen(val_X, val_Y, batch_size=batch_size)
    val_step_size = int(math.ceil(val_N / batch_size))

    # train
    model1.fit_generator(generator=train_gen,
                         epochs=300,
                         validation_data=val_gen,
                         steps_per_epoch=train_step_size,
                         validation_steps=val_step_size,
                         )
