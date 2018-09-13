import math


def pos_encode():
    pos = 1
    dim = 10

    pos_embeddings = []
    for i in range(dim):
        if i % 2 == 0:
            pos_embedding = math.cos(pos / 10000 ** (2 * i / dim))
        else:
            pos_embedding = math.sin(pos / 10000 ** (2 * i / dim))
        pos_embeddings.append(pos_embedding)
        print(pos_embedding)

pos_encode()