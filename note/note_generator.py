'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Finale module - network model on lead sheet generation given notes and chords
            as training data.

Improvements needed:
( ) - With autoencoder used, think how could the generation be done.
( ) - Should not use mean square error as autoencoder's loss. Change this.
      It causes problem to both beats data and note data.
( ) - We have the first result! But it is exceptionally awful. Optimize it!!
( ) - Data used is not enough. When code structure is more flexible, should incorporate more data.
'''
import os

from keras import Sequential
from keras.layers import Dense, RepeatVector, LSTM
from keras.models import load_model
from keras.utils import to_categorical
from tqdm import tqdm

from rhythm.rhythm_generator import RhythmGenerator
from rhythm.rhythm_extractor import RhythmExtractor
from chord.generator.chord_generator import ChordGenerator, CHORD_DICT
from chord.extractor.chord_extractor_generic import ChordExtractor
from music21 import *
import numpy as np
import matplotlib.pyplot as plt

MAX_BEAT_LENGTH = 1000
MAX_CHORD_LENGTH = 200
MAX_NOTE_LENGTH = 500
EPOCH_NUM = 100
FILENAME = '../dataset/ashover.abc'


class NoteGenerator():
    def __init__(self):
        print()

    def get_train_test_data(self, example, is_stacked=False):
        '''
        Feed in 1 example of abc object, return the train and test tensor data.
        Train data is defined by:   (0,1000) - beat vector, (1000, 1200) - chord vector,
                                    (1200, 1700) -  note vector
        :param example: 1 abc score object
        :return:
            stacked_input_pattern - shape (MAX_BEAT_LENGTH + MAX_CHORD_LENGTH, 1)
            one_hot_note_pattern - shape (MAX_NOTE_LENGTH, 13)

            both separated if is_stacked=False, else stacked (for autoencoder training).
        '''
        rhythm_extractor = RhythmExtractor(FILENAME)
        beat_pattern = rhythm_extractor.extract_rhythm_to_text(example)

        chord_extractor = ChordExtractor()
        chord_pattern = chord_extractor.extract_chords(example)

        # print(len([i for i in beat_pattern if i == '-']))
        chord_generator = ChordGenerator(FILENAME)
        one_hot_chord_pattern = chord_generator.one_hot_encode_chords(chord_pattern)
        chord_pattern = np.argmax(one_hot_chord_pattern, axis=1).reshape(-1, 1)

        rhythm_generator = RhythmGenerator(FILENAME)
        one_hot_beat_pattern = rhythm_generator.one_hot_encode_beats(beat_pattern)
        beat_pattern = np.argmax(one_hot_beat_pattern, axis=1).reshape(-1, 1)

        # Prepare input data - #beats + zeros + #chords + zeros
        stacked_input_pattern = np.append(beat_pattern, np.zeros(MAX_BEAT_LENGTH - len(beat_pattern)))
        stacked_input_pattern = np.append(stacked_input_pattern, chord_pattern)
        stacked_input_pattern = np.append(stacked_input_pattern, np.zeros(MAX_CHORD_LENGTH
                                                                          - len(chord_pattern))).reshape(-1, 1)

        note_pattern = chord_extractor.extract_measures_from_score(example)
        flatten_note_pattern = [note.replace('-', 'b') for measure in note_pattern
                                for note in measure]

        # chord is represented by note name, so we can use this to encode notes too
        # prepare output data - #notes + zeros
        encoded_note_pattern = np.array([CHORD_DICT[note] for note in flatten_note_pattern])
        encoded_note_pattern = np.append(encoded_note_pattern, np.zeros(MAX_NOTE_LENGTH -
                                                                        len(encoded_note_pattern)))
        one_hot_note_pattern = to_categorical(encoded_note_pattern, num_classes=13)

        if is_stacked:
            return np.append(stacked_input_pattern, encoded_note_pattern)
        else:
            return stacked_input_pattern, one_hot_note_pattern

    def build_model(self, X_train, X_test):
        '''
        Build a stacked autoencoder here.
        :param X_train
        :param X_test
        :return: model
        '''
        input_dim = X_train.shape[-1]

        # Start neural network
        model = Sequential()
        model.add(Dense(units=512, activation='relu', input_shape=(input_dim,)))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(input_dim))

        print(model.summary())
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, X_train, epochs=EPOCH_NUM)

        scores = model.evaluate(X_train, X_train, verbose=True)
        print('Train loss:', scores[0])
        print('Train accuracy:', scores[1])
        scores = model.evaluate(X_test, X_test, verbose=True)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        plt.plot(range(len(history.history['loss'])), history.history['loss'])
        plt.show()

        model.save('note_model.h5')
        return model

    def train_model(self):
        scores = converter.parse(FILENAME)
        if not os.path.exists('ashover-data.npy'):
            score_vectors = []
            note_vectors = []
            for score in tqdm(scores):
                score_vector, note_vector = self.get_train_test_data(score, is_stacked=True)
                score_vectors.append(score_vector)
                note_vectors.append(note_vector)
            score_vectors = np.array(score_vectors)
            note_vectors = np.array(note_vectors)
            np.save('ashover-data-data.npy', score_vectors)
        else:
            score_vectors = np.load('ashover-data.npy')

        if not os.path.exists('note_model.h5'):
            model = self.build_model(score_vectors[1].reshape(1,-1), score_vectors[1].reshape(1,-1))
        else:
            model = load_model('note_model.h5')

        # Predict the last score
        scores[1].show()
        predict_vector = np.squeeze(model.predict(
                            score_vectors[1].reshape(1, score_vectors[1].shape[0])))

        # post processing
        predict_vector = np.rint(predict_vector)
        predict_vector = self.predict_vector_check(predict_vector)

        plt.plot(range(len(score_vectors[1])), score_vectors[1], label='actual')
        plt.legend()
        plt.show()
        plt.plot(range(len(predict_vector)), predict_vector, label='predict')
        plt.legend()
        plt.show()

        # visualization
        self.convert_vector_to_score(predict_vector)

    def predict_vector_check(self, predict_vector):
        beat_anomaly = [i for i in predict_vector[:1000] if i < 0 or i > 2]
        chord_anomaly = [i for i in predict_vector[1000:1200] if i < 0 or i > 25]
        note_anomaly = [i for i in predict_vector[1200:1700] if i < 0 or i > 12]

        # print(beat_anomaly)
        # print(chord_anomaly)
        # print(note_anomaly)
        # print("Beat anomaly: {} Chord anomaly: {} Note anomaly: {}".format(len(beat_anomaly),
        #                                                                    len(chord_anomaly),
        #                                                                    len(note_anomaly)))

        def sanitize(predict_vector, upper_value, range_start, range_end):
            for i in range(range_start, range_end):
                if predict_vector[i] < 0:
                    predict_vector[i] = 0
                if predict_vector[i] > upper_value:
                    predict_vector[i] = upper_value
            return predict_vector

        predict_vector = sanitize(predict_vector, 2, 0, 1000)
        predict_vector = sanitize(predict_vector, 25, 1000, 1200)
        predict_vector = sanitize(predict_vector, 12, 1200, 1700)
        return predict_vector

    def convert_vector_to_score(self, vector):
        beat_vector = list(vector[:1000])
        note_vector = list(vector[1200:1700])

        # remove trailing zeros used for model training
        # problem: among the zeros there are some ones. problem comes from using m.s.e.
        cur_beat, cur_note = 0, 0
        while cur_beat == 0:
            cur_beat = beat_vector.pop()
        while cur_note == 0:
            cur_note = note_vector.pop()

        rhythm_generator = RhythmGenerator('')
        beat_pattern = rhythm_generator.decode_beats_from_rhythm_id(np.array(beat_vector))
        print(beat_pattern)
        rhythm_generator.show_beats_as_notes(beat_pattern, pitch_pattern=note_vector)

    def __convert_tensor_to_one_hot(self, predict_vector):
        one_hot_predict_vector = np.zeros_like(predict_vector)

        # TODO: optimize this, don't use for loop
        for i in range(len(one_hot_predict_vector)):
            # TODO: use random sampling, seems like the 1 note problem recurs again
            one_hot_predict_vector[i][np.argmax(predict_vector[i])] = 1

        return one_hot_predict_vector






def main():
    note_generator = NoteGenerator()
    note_generator.train_model()


if __name__ == "__main__":
    main()

