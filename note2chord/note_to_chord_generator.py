'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Note to chord generator.
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import keras.backend as K
from collections import Counter

from pretty_midi import pretty_midi

from utils import piano_roll_to_pretty_midi, chord_index_to_piano_roll

from keras.utils import to_categorical

from chord.chord_generator import ChordGenerator
from dataset.data_pipeline import DataPipeline
from models.model_builder import ModelBuilder
from utils import merge_melody_with_chords

MAX_NOTES = 1200


class NoteToChordGenerator:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None

    def generate_chords_from_note(self, notes=None, is_fast_load=False):
        model_name = "bidirectional"
        self.train_melody_to_chord_model(model_name=model_name, is_fast_load=is_fast_load)

        while len(notes) < MAX_NOTES:
            notes = np.pad(notes, (0, MAX_NOTES - len(notes)), mode='constant', constant_values=0)
        if model_name == "bidirectional":
            notes = to_categorical(notes, num_classes=128)
        if len(notes.shape) == 1:
            notes = np.expand_dims(notes, axis=-1)
        test_melody = np.expand_dims(notes, axis=0)

        # generate chords from notes
        result_chord_indices = np.argmax(self.model.predict(test_melody), axis=2)[0]
        result_chord_pr = chord_index_to_piano_roll(result_chord_indices)
        chord_midi = piano_roll_to_pretty_midi(result_chord_pr, fs=12)
        chord_midi.write('chords.mid')

        # test_melody = to_categorical(test_melody, num_classes=128)
        test_melody = np.transpose(np.squeeze(test_melody), (1,0))
        test_melody[test_melody > 0] = 90
        melody_midi = piano_roll_to_pretty_midi(test_melody, fs=12)
        melody_midi.write('melody.mid')

        # print(Counter(result_chord_indices))
        # merge together
        merge_melody_with_chords('melody.mid', 'chords.mid', 'song.mid')
        return result_chord_indices

    def train_melody_to_chord_model(self, tt_split=0.9, epochs=100, model_name='basic_rnn',
                                    is_fast_load=False):
        '''
        Train model step - model takes in melody piano roll and outputs chord piano roll.
        :param tt_split: train test split
        :param epochs:  number of epochs to train
        :param model_name: specify which model we are training
        :return: None. Model is assigned as self.model for this generator
        '''
        # clear session to avoid any errors
        K.clear_session()

        if not is_fast_load:
            # Train test split
            self.__prepare_data_tt_splited(tt_split=tt_split, model_name=model_name, src="nottingham-embed")
            input_dim = self.X_train.shape[1:]
            output_dim = self.Y_train.shape[-1]

        else:
            input_dim = (MAX_NOTES, 1)
            output_dim = 26

        # Load / train model
        if model_name == 'basic_rnn':
            model_file = "../note2chord/basic_rnn_750.h5"
            if os.path.exists(model_file):
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_basic_rnn_model(input_dim=input_dim,
                                                 output_dim=output_dim)
                model.load_weights(model_file)
            else:
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_basic_rnn_model(input_dim=input_dim,
                                                 output_dim=output_dim)
                model = mb.train_model(model, epochs, loss="categorical_crossentropy")
                model.save_weights(model_file)

        elif model_name == 'bidirectional':
            if not is_fast_load:
                self.X_train = to_categorical(self.X_train, num_classes=128)
                self.X_test = to_categorical(self.X_test, num_classes=128)
            else:
                input_dim = (MAX_NOTES, 128)

            model_file = "../note2chord/bidirectional_rnn.h5"
            if os.path.exists(model_file):
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_bidirectional_rnn_model_no_embeddings(input_dim=input_dim,
                                                 output_dim=output_dim)
                model.load_weights(model_file)
            else:
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_bidirectional_rnn_model_no_embeddings(input_dim=input_dim,
                                                 output_dim=output_dim)
                model = mb.train_model(model, epochs, loss="categorical_crossentropy")
                model.save_weights(model_file)

        elif model_name == "bidem":
            if os.path.exists("../note2chord/bidem.h5"):
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_bidirectional_rnn_model(input_dim=input_dim,
                                                         output_dim=output_dim)
                model.load_weights("bidem.h5")
            else:
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_bidirectional_rnn_model(input_dim=input_dim,
                                                         output_dim=output_dim)
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
    dp = DataPipeline()
    chords, melodies = dp.get_csv_nottingham_cleaned()
    result_indices = generator.generate_chords_from_note(melodies[1234])
    # print(result_indices)
    print(len(result_indices))