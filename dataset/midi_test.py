import os

from keras.models import load_model
from utils import piano_roll_to_pretty_midi
from dataset.data_pipeline import DataPipeline
from models.model_builder import ModelBuilder
import numpy as np
import matplotlib.pyplot as plt


def main():
    dp = DataPipeline()
    chords, melodies = dp.get_nottingham_piano_roll()
    chords[chords > 0] = 1
    melodies[melodies > 0] = 1
    csparsity = 1.0 - np.count_nonzero(chords) / chords.size
    msparsity = 1.0 - np.count_nonzero(melodies) / melodies.size
    print(csparsity, msparsity)
    cshape, mshape = chords.shape, melodies.shape

    # for basic rnn
    chords, melodies = chords.reshape(cshape[0], cshape[1] * cshape[3], cshape[2]), \
                       melodies.reshape(mshape[0], mshape[1] * mshape[3], mshape[2])
    # melodies = np.expand_dims(np.argmax(melodies, axis=-1), axis=-1)
    # chords = np.expand_dims(chords, axis=-1)
    # melodies = melodies.reshape(mshape[0], mshape[1], mshape[2] * mshape[3])
    unique, counts = np.unique(chords, return_counts=True)
    print(dict(zip(unique, counts)))
    print(chords.shape, melodies.shape)
    tt_split = 0.9
    split_ind = int(tt_split * len(chords))
    X_train, Y_train, X_test, Y_test = chords[:split_ind], melodies[:split_ind], \
                                       chords[split_ind:], melodies[split_ind:]

    if os.path.exists("basic_rnn.h5"):
        basic_rnn_model = load_model("basic_rnn.h5")
    else:
        mb = ModelBuilder(X_train, Y_train, X_test, Y_test)
        basic_rnn_model = mb.build_basic_rnn_model(input_dim=X_train.shape[1:])
        basic_rnn_model = mb.train_model(basic_rnn_model, 10, loss="binary_crossentropy")
        basic_rnn_model.save("basic_rnn.h5")

    ind = 169
    # y = basic_rnn_model.predict(X_train[ind].reshape(1, 100, 128, 12, 1))
    y = basic_rnn_model.predict(X_train[ind].reshape(1, 1200, 128))
    print(y.shape)
    # print(y[0][:,0])
    plt.imshow(y[0], cmap='gray')
    plt.show()

    # y[0] = -y[0]
    y = abs(y)
    plt.hist(y.reshape(-1))
    plt.show()

    y[y > 0.3] = 90
    y[y <= 0.3] = 0
    plt.imshow(Y_train[ind])
    plt.show()
    plt.imshow(y[0])
    plt.show()

    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(Y_train[ind], return_counts=True)
    print(dict(zip(unique, counts)))
    print(y.shape)
    y = y[0].reshape(128, -1)
    actual = Y_train[ind].reshape(128, -1)
    actual[actual > 0] = 90

    y_midi = piano_roll_to_pretty_midi(y, fs=12)
    y_midi.write('test.mid')
    actual = piano_roll_to_pretty_midi(actual, fs=12)
    actual.write('actual.mid')

def main2():
    dp = DataPipeline()
    chords, melodies = dp.get_nottingham_piano_roll()
    chords[chords > 0] = 1
    melodies[melodies > 0] = 1
    cshape, mshape = chords.shape, melodies.shape
    print(cshape, mshape)

if __name__ == "__main__":
    main()