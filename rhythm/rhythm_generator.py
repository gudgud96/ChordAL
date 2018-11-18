'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    - Train LSTM model on extracted and encoded rhythm data
            - Generate rhythm beats from encoded data by prediction within a 8-quaver timeframe

Improvements needed:
(/) Fix train-test split as multiple pieces of a certain timeframe
(/) Fix generate_beats (predict the next beat over a timeframe iteratively)
(/) The pattern just repeats on its own! Solve this! - use probability distribution sampling
( ) Analyze the generated beats. Seems to have no sense of bars and time signature
( ) When the project grows, this script should be written in object-oriented style
( ) Haven't encounter for non-4/4 time signature, semi-quavers, upbeats
( ) LSTM training time-frame is 8-quaver timeframe, note by note generation
'''
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical 
from keras.models import load_model
from music21 import *
import numpy as np
import os.path
import random
import chord.generator.chord_generator as chord_generator

# Configurable variables
BEATS_FILE = "beats_nottingham_training.txt"
BEATS_PATTERN_LENGTH = 535
TT_SPLIT = 0.8  # percentage of train / test
CHAR_DICT = {
    "-": 1,     # hit note
    ">": 2,     # hold note
    "_": 0      # rest note
}
DECODE_DICT = {v: k for k, v in CHAR_DICT.items()}
TIME_STEP = 0.25  # 16th note for our time step
TIME_FRAME = 17   # 16 notes as pre-loaded timeframe for training, 1 for prediction
NUM_CLASSES = len(CHAR_DICT.keys())


class RhythmGenerator:
    def __init__(self, train_file):
        self.train_file = train_file

    def preprocess_data(self, tt_split = 0.8):
        '''
        Preprocess incoming data given filename.
        :param tt_split: train-test split percentage
        :return: X_train, X_test, Y_train, Y_test
        '''
        filename = self.train_file
        beats_lines = [line.rstrip() for line in open(filename, 'r+')]
        beats_processed = np.zeros((len(beats_lines), len(beats_lines[0])))
        for i in range(len(beats_processed)):
            for j in range(len(beats_processed[0])):
                beats_processed[i][j] = CHAR_DICT[beats_lines[i][j]]    # enumeration using dict

        # Generate train and test set
        # Idea: specify a timeframe (set as 16 for now) and predict the next beat after 16 beats
        # Slice each song into pieces of length 17, 16 to training set and 1 to test set
        # Problem with this idea: you may start with '>', which seems a bit absurd, but let's see how

        beats_in_tf = []
        # print(len(beats_processed))
        # print(len(beats_processed[0]))
        for i in range(len(beats_processed)):
            cur_beat, cur_beat_len = beats_processed[i], len(beats_processed[i])
            for j in range(cur_beat_len - TIME_FRAME + 1):
                beats_in_tf.append([cur_beat[j : j + TIME_FRAME]])

        beats_in_tf = np.squeeze(np.asarray(beats_in_tf))
        print("beats_in_tf.shape : {}".format(beats_in_tf.shape))
        X, Y = beats_in_tf[:, :-1], beats_in_tf[:, -1]
        X_oh, Y_oh = to_categorical(X, num_classes=NUM_CLASSES), to_categorical(Y, num_classes=NUM_CLASSES)

        tt_split_index = int(tt_split * len(beats_in_tf))
        X_train, X_test, Y_train, Y_test = X_oh[:tt_split_index], X_oh[tt_split_index:], \
                                           Y_oh[:tt_split_index],Y_oh[tt_split_index:]

        print("X_train.shape: {}, Y_train.shape: {}".format(X_train.shape, Y_train.shape))
        print("X_test.shape:{} , Y_test.shape: {}".format(X_test.shape, Y_test.shape))
        return X_train, X_test, Y_train, Y_test

    def build_model(self, X_train, X_test, Y_train, Y_test):
        '''
        Build neural network model here.
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
        model.fit(X_train, Y_train, epochs=50)

        scores = model.evaluate(X_train, Y_train, verbose=True)
        print('Train loss:', scores[0])
        print('Train accuracy:', scores[1])
        scores = model.evaluate(X_test, Y_test, verbose=True)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        return model

    def generate_next_beat(self, beat_block, model):
        '''
        Generate next beat given a beat block.
        :param beat_block
        :param model
        :return: 1 predicted beat
        '''
        # print('Incoming: ' + beat_block)
        predict_beat_oh = self.one_hot_encode_beats(beat_block)
        # print(predict_beat_oh.shape)
        prediction = model.predict_classes(predict_beat_oh.reshape(1, predict_beat_oh.shape[0], predict_beat_oh.shape[1]))
        prediction_proba = model.predict(predict_beat_oh.reshape(1, predict_beat_oh.shape[0], predict_beat_oh.shape[1]))
        prediction_random_sample = np.random.choice(3, 1, p=prediction_proba.flatten())[0]
        # print('Generated: ' + decode_beats_from_one_hot(prediction))
        return self.decode_beats_from_rhythm_id(prediction_random_sample)      # random sampling
        # return decode_beats_from_one_hot(prediction)                  # no random sampling

    def generate_beats_from_seed(self, example_beat, model, show_beats=False):
        '''
        Generate beats given example beats data
        :param model
        :return: result_beat
        '''
        # example_beat = open(BEATS_FILE).readlines()[0]
        seed_beat = example_beat[:TIME_FRAME]    # seed beat pattern for generation
        result_beat = seed_beat[:]
        index = 0
        while len(result_beat) < BEATS_PATTERN_LENGTH and result_beat[-4:] != '____':   # terminating conditions
            cur_beat_block = result_beat[index : index + TIME_FRAME - 1]
            result_beat += self.generate_next_beat(cur_beat_block, model)
            index += 1
            if result_beat[-4:] == '____':
                print('Breaking the 2nd rule')
        open('example_beat.txt', 'w+').write(example_beat)
        open('result_beat.txt', 'w+').write(result_beat)
        if show_beats:
            self.show_beats_as_notes(example_beat)
            self.show_beats_as_notes(result_beat)

        return result_beat

    def one_hot_encode_beats(self, beat_pattern):
        '''
        Encode beat pattern to one-hot matrix.
        :return: one-hot matrix of beat sequence
        '''
        encoded_beat_pattern = np.zeros((len(beat_pattern)))
        for i in range(len(beat_pattern)):
            encoded_beat_pattern[i] = CHAR_DICT[beat_pattern[i]]
        return to_categorical(encoded_beat_pattern, num_classes=NUM_CLASSES)

    def decode_beats_from_rhythm_id(self, rhythm_id_beats):
        '''
        Decode matrix of rhythm id sequence to beats.
        :return: beat sequence in characters
        '''
        return ''.join([DECODE_DICT[encode] for encode in rhythm_id_beats.flatten()])

    def show_beats_as_notes(self, beat_pattern, pitch_pattern=None):
        '''
        Helper function to show beats as notes. For analysis purpose.
        '''
        cur_quarter_len = 0
        note_stream = stream.Stream()

        pitch = pitch_pattern.pop(0) if pitch_pattern else None

        def create_note(note_stream, cur_quarter_len, pitch, pitch_pattern):
            note_stream, cur_quarter_len = self.create_note_to_stream(
                    note_stream, cur_quarter_len, is_note='note', pitch=pitch)
            next_pitch = pitch_pattern.pop(0) if pitch_pattern else None

            return note_stream, cur_quarter_len, next_pitch, pitch_pattern

        for i in range(len(beat_pattern)):
            cur_char = beat_pattern[i]

            if cur_char == '-':     # hit note
                if cur_quarter_len > 0:
                    note_stream, cur_quarter_len, pitch, pitch_pattern = \
                        create_note(note_stream, cur_quarter_len, pitch, pitch_pattern)

                cur_quarter_len += TIME_STEP
                if i == len(beat_pattern) - 1:
                    note_stream, cur_quarter_len, pitch, pitch_pattern = \
                        create_note(note_stream, cur_quarter_len, pitch, pitch_pattern)

            elif cur_char == '>':   # hold note
                cur_quarter_len += TIME_STEP
                if i == len(beat_pattern) - 1:
                    note_stream, cur_quarter_len, pitch, pitch_pattern = \
                        create_note(note_stream, cur_quarter_len, pitch, pitch_pattern)

            else:                   # rest note
                if beat_pattern[i-1] != '_':
                    note_stream, cur_quarter_len, pitch, pitch_pattern = \
                        create_note(note_stream, cur_quarter_len, pitch, pitch_pattern)

                cur_quarter_len += TIME_STEP
                if i == len(beat_pattern) - 1 or i < len(beat_pattern) - 1 \
                                                and beat_pattern[i+1] != '_':
                    note_stream, cur_quarter_len = self.create_note_to_stream(
                        note_stream, cur_quarter_len, is_note='rest', pitch=pitch)

        note_stream.show()

    def create_note_to_stream(self, stream, quarter_len, is_note='note', pitch=None):
        '''
        Helper function to create note to stream.
        '''
        if is_note == 'note':
            if not pitch:
                new_note = note.Note()
                new_note.pitch.pitchClass = random.randint(0, 15)
            else:
                note_name = chord_generator.DECODE_DICT[pitch] + '4'
                new_note = note.Note(note_name)

        else:
            new_note = note.Rest()
        new_note.duration = duration.Duration(quarter_len)
        stream.append(new_note)
        return stream, 0        # reset cur_quarter_len to 0

    def generate_beats(self, example_index=0, is_retrain=False):
        '''
        Main method to generate beats.
        :param example_index:
        :param is_retrain:
        :return: Generated beats
        '''
        X_train, X_test, Y_train, Y_test = self.preprocess_data(tt_split=TT_SPLIT)
        if os.path.exists('rhythm_model.h5') and is_retrain == False:
            print('Loading rhythm_model.h5...')
            model = load_model('rhythm_model.h5')
        else:
            model = self.build_model(X_train, X_test, Y_train, Y_test)
            model.save('rhythm_model.h5')

        example_beat = open(self.train_file).readlines()[example_index]
        return self.generate_beats_from_seed(example_beat, model)


if __name__ == "__main__":
    rhythm_generator = RhythmGenerator(train_file=BEATS_FILE)
    generated_beats = rhythm_generator.generate_beats()
    print(generated_beats)