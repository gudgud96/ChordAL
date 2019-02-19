'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Note to chord generator.
'''
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

from pretty_midi import pretty_midi

from utils import piano_roll_to_pretty_midi, chord_index_to_piano_roll

from keras.utils import to_categorical

from chord.chord_generator import ChordGenerator
from dataset.data_pipeline import DataPipeline
from models.model_builder import ModelBuilder
from generator.song_generator import merge_melody_with_chords


class NoteToChordGenerator:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None

    def generate_chords_from_note(self):
        self.train_melody_to_chord_model(model_name="basic_rnn")
        test_melody = np.expand_dims(self.X_train[0], axis=0)

        # generate chords from notes
        result_chord_indices = np.argmax(self.model.predict(test_melody), axis=2)[0]
        result_chord_pr = chord_index_to_piano_roll(result_chord_indices)
        chord_midi = piano_roll_to_pretty_midi(result_chord_pr, fs=12)
        chord_midi.write('chords.mid')

        # just for reference
        actual_chord_indices = np.argmax(self.Y_train[0], axis=1)
        actual_chord_pr = chord_index_to_piano_roll(actual_chord_indices)
        actual_chord_midi = piano_roll_to_pretty_midi(actual_chord_pr, fs=12)
        actual_chord_midi.write('actual_chords.mid')

        # get the melody
        test_melody = to_categorical(self.X_train[0], num_classes=128)
        test_melody = np.transpose(np.squeeze(test_melody), (1,0))
        test_melody[test_melody > 0] = 90
        melody_midi = piano_roll_to_pretty_midi(test_melody, fs=12)
        melody_midi.write('melody.mid')

        print(Counter(result_chord_indices))
        # merge together
        merge_melody_with_chords('melody.mid', 'chords.mid', 'song.mid')
        merge_melody_with_chords('melody.mid', 'actual_chords.mid', 'actual_song.mid')

    def train_melody_to_chord_model(self, tt_split=0.9, epochs=100, model_name='basic_rnn'):
        '''
        Train model step - model takes in melody piano roll and outputs chord piano roll.
        :param tt_split: train test split
        :param epochs:  number of epochs to train
        :param model_name: specify which model we are training
        :return: None. Model is assigned as self.model for this generator
        '''

        # Train test split
        self.__prepare_data_tt_splited(tt_split=tt_split, model_name=model_name, src="nottingham-embed")
        # print('Chords shape: {}  Melodies shape: {}'.format(chords.shape, melodies.shape))

        # Load / train model
        if model_name == 'basic_rnn':
            model_file = "basic_rnn.h5"
            if os.path.exists(model_file):
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_basic_rnn_model(input_dim=self.X_train.shape[1:],
                                                 output_dim=self.Y_train.shape[-1])
                model.load_weights(model_file)
            else:
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_basic_rnn_model(input_dim=self.X_train.shape[1:],
                                                 output_dim=self.Y_train.shape[-1])
                model = mb.train_model(model, epochs, loss="categorical_crossentropy")
                model.save_weights(model_file)

        elif model_name == "bidem":
            if os.path.exists("bidem.h5"):
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_bidirectional_rnn_model(input_dim=self.X_train.shape[1:],
                                                         output_dim=self.Y_train.shape[-1])
                model.load_weights("bidem.h5")
            else:
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_bidirectional_rnn_model(input_dim=self.X_train.shape[1:],
                                                         output_dim=self.Y_train.shape[-1])
                model = mb.train_model(model, epochs, loss="categorical_crossentropy")
                model.save_weights("bidem.h5")

        self.model = model
        print("Model {} loaded.".format(model_name))

    def __get_raw_data(self, src='nottingham', model_name='basic_rnn'):
        '''
        Get raw data depending on data source and model.
        :param src: Data source includes 'nottingham' and 'lakh'.
        :param model_name: Model includes 'basic_rnn'.
        :return:
        '''
        if src == 'nottingham':
            dp = DataPipeline()
            chords, melodies = dp.get_nottingham_piano_roll(is_small_set=True, is_shifted=False)

            chords[chords > 0] = 1
            melodies[melodies > 0] = 1
            csparsity = 1.0 - np.count_nonzero(chords) / chords.size
            msparsity = 1.0 - np.count_nonzero(melodies) / melodies.size
            cshape, mshape = chords.shape, melodies.shape
            chords, melodies = self.__process_raw_data(chords, melodies, model=model_name)

        elif src == 'nottingham-embed':
            dp = DataPipeline()
            chords, melodies = dp.get_nottingham_embed(is_small_set=True)
            melodies[melodies > 0] = 1
            print(chords.shape, melodies.shape)

        return chords, melodies

    def __process_raw_data(self, chords, melodies, model='basic_rnn'):
        melodies = np.expand_dims(np.argmax(melodies, axis=1), axis=-1)
        chords = to_categorical(chords, num_classes=26)
        return chords, melodies

    def __prepare_data_tt_splited(self, tt_split=0.9, src='nottingham', model_name='basic_rnn'):
        chords, melodies = self.__get_raw_data(src=src, model_name=model_name)
        chords, melodies = self.__process_raw_data(chords, melodies)

        print("After preprocessing: ", chords.shape, melodies.shape)

        # Melodies as train data, chords as test data
        split_ind = int(tt_split * len(chords))
        self.X_train, self.Y_train, self.X_test, self.Y_test = melodies[:split_ind], \
                                                               chords[:split_ind], \
                                                               melodies[split_ind:], \
                                                               chords[split_ind:]


if __name__ == "__main__":
    generator = NoteToChordGenerator()
    generator.generate_chords_from_note()