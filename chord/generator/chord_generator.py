'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Train and generate chord sequence from chord sequence dataset.

Improvements needed:
(/) Data preprocess done. Build model for training and generation.
( ) Find a way to train for more epochs. Now only train 5 epochs with 80+% accuracy.
( ) Do an analysis on the usage of chords at the commented section.
( ) Duplicate code with rhythm generator may need refactoring in preprocess_data and model building.

'''
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import os


# Configurable variables
CHORD_SEQUENCE_FILE = "chord_sequence_file.txt"
TT_SPLIT = 0.8  # percentage of train / test
CHORD_DICT = {
  "Cb": 12, "C": 1, "C#": 2, "Db": 2, "D": 3, "D#": 4, "Eb": 4, "E": 5, "E#": 6,
  "Fb": 5, "F": 6, "F#": 7, "Gb": 7, "G": 8, "G#": 9, "Ab": 9, "A": 10, "A#": 11, 
  "Bb": 11, "B": 12, "B#": 1
}
DECODE_DICT = {
  1: "C", 2: "C#", 3: "D", 4: "D#", 5: "E", 6: "F", 7: "F#", 8: "G", 9: "G#", 10: "A", 11: "A#", 12: "B"
}
MAJ, MIN = ["maj", "min"]   # major is even and minor is +1 odd. Hence Cmaj=2, Cmin=3, C#maj=4 and etc.
TIME_FRAME = 9              # 8 notes as pre-loaded timeframe for training, 1 for prediction
NUM_CLASSES = 26            # 2-25 for major and minor, 0 for None, 1 is not used
EPOCH_NUM = 5               # number of epochs for training

class ChordGenerator:
    def __init__(self, filename):
        self.train_file = filename

    def preprocess_data(self, tt_split = 0.8):
        '''
        Preprocess data.
        :param tt_split: train test split percentage
        :return: X_train, X_test, Y_train, Y_test
        '''
        filename = self.train_file
        chord_sequences = [a.rstrip() for a in open(filename, 'r+').readlines()]
        len_list = [len(s.split(' > ')) for s in chord_sequences]
        # plt.hist(len_list, normed=True, bins=30)    # from this histogram, median is 210 and max is 400+. take 300 as max.
        # plt.show()
        max_len = 300
        chords_processed = np.zeros((len(chord_sequences), max_len))

        for i in range(len(chord_sequences)):
            sequence = chord_sequences[i]
            chords = sequence.split(' > ')
            chords = chords[:] if len(chords) <= max_len else chords[:max_len]
            for j in range(len(chords)):
                chords_processed[i][j] = self.chord_to_id(chords[j])

        # print(chord_sequences[12])
        # print(chords_processed[12])

        # TODO: do an analysis of usage of chords

        # same strategy as rhythm generator, but tf=8
        # needs to reconsider if this is a good strategy, because global structure will be lost.
        chords_in_tf = []
        for i in range(len(chords_processed)):
            cur_chords, cur_chords_len = chords_processed[i], len(chords_processed[i])
            for j in range(cur_chords_len - TIME_FRAME + 1):
                chords_in_tf.append([cur_chords[j : j + TIME_FRAME]])

        chords_in_tf = np.squeeze(np.asarray(chords_in_tf))
        # print(chords_in_tf[:10])
        print("chords_in_tf.shape : {}".format(chords_in_tf.shape))
        X, Y = chords_in_tf[:, :-1], chords_in_tf[:, -1]
        X_oh, Y_oh = to_categorical(X, num_classes=NUM_CLASSES), to_categorical(Y, num_classes=NUM_CLASSES)

        tt_split_index = int(tt_split * len(chords_in_tf))
        X_train, X_test, Y_train, Y_test = X_oh[:tt_split_index], X_oh[tt_split_index:], \
                                           Y_oh[:tt_split_index],Y_oh[tt_split_index:]

        print("X_train.shape: {}, Y_train.shape: {}".format(X_train.shape, Y_train.shape))
        print("X_test.shape:{} , Y_test.shape: {}".format(X_test.shape, Y_test.shape))
        return X_train, X_test, Y_train, Y_test

    def build_model(self, X_train, X_test, Y_train, Y_test):
        '''
        Build the neural network model.
        :param X_test
        :param Y_train
        :param Y_test
        :return: model
        '''
        num_seq, num_dim = X_train.shape[1], X_train.shape[2]

        model = Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=(num_seq, num_dim)))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(num_dim))
        model.add(Activation('softmax'))

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=EPOCH_NUM)

        scores = model.evaluate(X_train, Y_train, verbose=True)
        print('Train loss:', scores[0])
        print('Train accuracy:', scores[1])
        scores = model.evaluate(X_test, Y_test, verbose=True)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        return model

    def generate_chord_from_seed(self, example_chord, model, num_of_chords=40):
        '''
        Generate chord given example.
        :param model:
        :param num_of_chords: Number of chords needed.
        :return:
        '''
        # example_chord = open(CHORD_SEQUENCE_FILE).readlines()[1].split(' > ')
        seed_chord = example_chord[:TIME_FRAME - 1]    # seed beat pattern for generation
        result_chord = seed_chord[:]
        index = 0
        while len(result_chord) < num_of_chords:   # terminating condition now only consist of a fixed length
            cur_chord_block = result_chord[index : index + TIME_FRAME - 1]
            next_chord = self.generate_next_chord(cur_chord_block, model)
            result_chord.append(next_chord)
            index += 1
        open('example_chord.txt', 'w+').write(' > '.join(example_chord))
        open('result_chord.txt', 'w+').write(' > '.join(result_chord))
        print(example_chord)
        print(result_chord)

        return result_chord

    def generate_next_chord(self, chord_block, model):
        '''
        Generate next chord given a chord block.
        :param chord_block:
        :param model:
        :return: next chord
        '''
        predict_chord_oh = self.one_hot_encode_chords(chord_block)
        # print(predict_chord_oh)
        prediction = model.predict_classes(predict_chord_oh.reshape(1, predict_chord_oh.shape[0], predict_chord_oh.shape[1]))
        prediction_proba = model.predict(predict_chord_oh.reshape(1, predict_chord_oh.shape[0], predict_chord_oh.shape[1]))
        prediction_random_sample = np.random.choice(NUM_CLASSES, 1, p=prediction_proba.flatten())[0]
        # print('Generated: ' + id_to_chord(prediction_random_sample))
        print("Random sample: {}".format(prediction_random_sample))
        return self.id_to_chord(prediction_random_sample)      # random sampling
        # return decode_beats_from_one_hot(prediction)                  # no random sampling

    def one_hot_encode_chords(self, chord_block):
        '''
        Helper function to encode chord block to one-hot matrix.
        :return: one-hot martix of chord
        '''
        if isinstance(chord_block, str):
            chord_block = chord_block.split(' > ')
        encoded_chord_pattern = np.zeros((len(chord_block)))
        for i in range(len(chord_block)):
            encoded_chord_pattern[i] = self.chord_to_id(chord_block[i])
        return to_categorical(encoded_chord_pattern, num_classes=NUM_CLASSES)

    def decode_chords_from_one_hot(self, one_hot_beats):
        return ''.join([DECODE_DICT[encode] for encode in one_hot_beats.flatten()])

    # change chord of format <name>:<tonality> to id between 2-25
    def chord_to_id(self, chord):
        if ':' not in chord:
            return 0
        chord_name, chord_tonality = chord.split(':')
        chord_id = CHORD_DICT[chord_name] * 2           # leave 0 for empty chords
        if MIN in chord_tonality:
            chord_id += 1                               # odd number for minors
        return chord_id

    def id_to_chord(self, idnum):
        if idnum == 0:
            return "-"
        elif idnum % 2 == 0:
            return DECODE_DICT[idnum / 2] + ':' + MAJ
        else:
            return DECODE_DICT[(idnum - 1) / 2] + ':' + MIN

    def generate_chords(self, example_index=0, num_of_chords=40, is_retrain=False):
        X_train, X_test, Y_train, Y_test = self.preprocess_data(self.train_file, TT_SPLIT)
        if os.path.exists('chord_model.h5') and is_retrain==False:
            print('Loading chord_model.h5...')
            model = load_model('chord_model.h5')
        else:
            model = self.build_model(X_train, X_test, Y_train, Y_test)
            model.save('chord_model.h5')

        example_chord = open(self.train_file).readlines()[example_index].split(' > ')
        self.generate_chord_from_seed(example_chord, model, num_of_chords)


if __name__ == "__main__":
    chord_generator = ChordGenerator(CHORD_SEQUENCE_FILE)
    chord_generator.generate_chords()
