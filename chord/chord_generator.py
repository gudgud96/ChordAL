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
import random

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import TimeDistributed
from music21 import *
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import os


# Configurable variables
CHORD_SEQUENCE_FILE = "../chord/chord_sequence_all_no_repeat.txt"
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
TIME_FRAME = 8              # 8 notes timeframe, train by sliding window
NUM_CLASSES = 26            # 2-25 for major and minor, 0 for None, 1 is not used
EPOCH_NUM = 5               # number of epochs for training


class ChordGenerator:
    def __init__(self, filename):
        self.train_file = filename

    def preprocess_data(self, tt_split = 0.9, is_small_dataset=True):
        '''
        Preprocess data.
        :param tt_split: train test split percentage
        :return: X_train, X_test, Y_train, Y_test
        '''
        filename = self.train_file
        chord_sequences = [a.rstrip() for a in open(filename, 'r+').readlines()]
        random.shuffle(chord_sequences)
        if is_small_dataset:
            chord_sequences = chord_sequences[:2000]

        # len_list = [len(k.split(' > ')) for k in chord_sequences]
        # plt.hist(len_list, normed=True, bins=10)
        # from this histogram, median is 50. take 50 as max.
        # plt.show()

        max_len = 50
        chords_processed = np.zeros((len(chord_sequences), max_len))

        for i in range(len(chord_sequences)):
            sequence = chord_sequences[i]
            chords = sequence.split(' > ')
            chords = chords[:] if len(chords) <= max_len else chords[:max_len]
            for j in range(len(chords)):
                chords_processed[i][j] = self.chord_to_id(chords[j])

        # pad chords_processed with TIME_FRAME - 1 zeros, so we can generate
        # starting with chords lesser than TIME_FRAME
        chords_processed = np.pad(chords_processed, ((0,0), (7,0)), 'constant', constant_values=0)

        # print(chord_sequences[12])
        # print(chords_processed[12])

        # same strategy as rhythm generator, but tf=8, many-to-one
        chords_in_tf = []
        for i in range(len(chords_processed)):
            cur_chords, cur_chords_len = chords_processed[i], len(chords_processed[i])
            for j in range(cur_chords_len - TIME_FRAME):
                chords_in_tf.append([cur_chords[j : j + TIME_FRAME + 1]])

        chords_in_tf = np.squeeze(np.asarray(chords_in_tf))
        print("chords_in_tf.shape : {}".format(chords_in_tf.shape))
        X, Y = chords_in_tf[:, :-1], chords_in_tf[:, -1]
        X_oh, Y_oh = to_categorical(X, num_classes=NUM_CLASSES), to_categorical(Y, num_classes=NUM_CLASSES)

        tt_split_index = int(tt_split * len(chords_in_tf))
        X_train, X_test, Y_train, Y_test = X_oh[:tt_split_index], X_oh[tt_split_index:], \
                                           Y_oh[:tt_split_index],Y_oh[tt_split_index:]

        # print("X_train.shape: {}, Y_train.shape: {}".format(X_train.shape, Y_train.shape))
        # print("X_test.shape:{} , Y_test.shape: {}".format(X_test.shape, Y_test.shape))
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
        model.add(LSTM(64, return_sequences=True, input_shape=(num_seq, num_dim)))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(LSTM(128, return_sequences=False))
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

    def __generate_chord_from_seed(self, example_chord, model, num_of_chords=32):
        '''
        Private method to generate chords given a seed sequence of chords
        :param example_chord:
                A seed sequence of chords of any length to kick start the generation.
        :param model:
                Model used to generate the chords.
        :param num_of_chords:
                Number of chords needed for generation.
        :return:
                chord sequence generated
        '''
        seed_chord = example_chord[:]    # seed beat pattern for generation
        # print('seed chord', seed_chord)
        result_chord = seed_chord[:]
        index = 0

        while len(result_chord) < num_of_chords:   # terminating condition now only consist of a fixed length
            cur_chord_block = result_chord[index : index + TIME_FRAME]
            next_chord = self.__generate_next_chord(cur_chord_block, model)
            result_chord.append(next_chord)
            index += 1

        while result_chord[0] == '-':
            cur_chord_block = result_chord[len(result_chord) - TIME_FRAME:]
            next_chord = self.__generate_next_chord(cur_chord_block, model)
            result_chord.append(next_chord)
            result_chord.pop(0)

        open('example_chord.txt', 'w+').write(' > '.join(example_chord))
        open('result_chord.txt', 'w+').write(' > '.join(result_chord))

        return result_chord

    def __generate_next_chord(self, chord_block, model):
        '''
        Private method to generate next chord given a chord block.
        :param chord_block:
        :param model:
        :return: next chord
        '''
        predict_chord_oh = self.one_hot_encode_chords(chord_block)
        predict_chord_oh = np.expand_dims(predict_chord_oh, axis=0)

        # prediction = model.predict_classes(predict_chord_oh)
        prediction_proba = model.predict(predict_chord_oh)
        prediction_random_sample = np.random.choice(NUM_CLASSES, 1, p=prediction_proba.flatten())[0]
        result = self.id_to_chord(prediction_random_sample)      # random sampling
        return result

    def chords_to_midi(self, chords):
        s = stream.Stream()
        for c in chords:
            if c == '-':
                continue        # TODO: Not a good strategy
            note, quality = c.split(':')
            note_value = CHORD_DICT[note]
            if quality == 'maj':
                value_lst = [note_value, (note_value + 4) % 12, (note_value + 7) % 12]
            else:
                value_lst = [note_value, (note_value + 3) % 12, (note_value + 7) % 12]

            for i in range(len(value_lst)):
                if value_lst[i] == 0:
                    value_lst[i] = 12

            if value_lst[1] > value_lst[0]:
                if value_lst[2] > value_lst[1]:
                    octave_lst = [3,3,3]
                else:
                    octave_lst = [3,3,4]
            else:
                octave_lst = [2,3,3]

            new_value_lst = []
            for i in range(len(value_lst)):
                new_value_lst.append(DECODE_DICT[value_lst[i]] + str(octave_lst[i]))

            d = duration.Duration(4)    # whole note length
            temp_chord = chord.Chord(new_value_lst, duration=d)
            s.append(temp_chord)

        fp = 'chords_to_midi.mid'
        s.write('midi', fp=fp)
        return fp

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

    def generate_chords(self, example_chord=None, num_of_chords=32, is_retrain=False):
        X_train, X_test, Y_train, Y_test = self.preprocess_data(TT_SPLIT, is_small_dataset=False)
        if os.path.exists('../chord/chord_model.h5') and not is_retrain:
            print('Loading chord_model.h5...')
            model = load_model('../chord/chord_model.h5')
        else:
            model = self.build_model(X_train, X_test, Y_train, Y_test)
            model.save('../chord/chord_model.h5')

        if not example_chord:
            # if no example chord sequence is provided, randomly generate one from train data
            index = random.randint(0, 100)
            example_chord = open(self.train_file).readlines()[index].split(' > ')
            example_chord = example_chord[:8]

        else:
            # if given example chord sequence length < TIME_FRAME, pad with zeros
            if len(example_chord) < TIME_FRAME:
                for i in range(TIME_FRAME - len(example_chord)):
                    example_chord.insert(0, '-')
            else:
                example_chord = example_chord[:TIME_FRAME]

        return self.__generate_chord_from_seed(example_chord, model, num_of_chords)

    def evaluate_chord_variation_coefficient(self, chords):
        '''
        Simple evaluations on chords generated.
        :param chords:
            Chords sequence generated.
        :return:
            No return, but 3 metrics printed -
                Chord variation - count dict of the chord sequence
                Chord variation coefficient - 1 - (# unique chords / # total chords)
                Chord distribution variance - Variance of counts for each chord
        '''
        chords = [self.chord_to_id(chord) for chord in chords]
        chords = [self.id_to_chord(id) for id in chords]
        count_dict = {}
        for chord in chords:
            if chord not in count_dict:
                count_dict[chord] = chords.count(chord)
        print("Chord variation: {}".format(count_dict))
        print("Chord variation coefficient: {}".format(1 - len(list(count_dict.keys())) / len(chords)))
        print("Chord distribution variance: {}".format(np.var(list(count_dict.values()))))


if __name__ == "__main__":
    chord_generator = ChordGenerator(CHORD_SEQUENCE_FILE)
    result = chord_generator.generate_chords(['G:min', 'D:maj', 'Eb:maj'])
    print(result)
    chord_generator.evaluate_chord_variation_coefficient(result)
    chord_generator.chords_to_midi(result)