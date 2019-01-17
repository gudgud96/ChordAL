'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Chord to note generator.

Improvements needed:
(/) - Include Basic RNN model.
'''
import sys,os
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))

import os
from keras.models import load_model
from utils import piano_roll_to_pretty_midi
from dataset.data_pipeline import DataPipeline
from models.model_builder import ModelBuilder
import numpy as np
import matplotlib.pyplot as plt


class ChordToNoteGenerator:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None

    def train_chord_to_melody_model(self, tt_split=0.9, epochs=100, model_name='basic_rnn'):
        '''
        Train model step - model takes in chord piano roll and outputs melody piano roll.
        :param tt_split: train test split
        :param epochs:  number of epochs to train
        :param model_name: specify which model we are training
        :return: None. Model is assigned as self.model for this generator
        '''
        # Load data
        # chords, melodies = self.__get_raw_data()
        # chords, melodies = self.__process_raw_data(chords, melodies)
        # unique, counts = np.unique(chords, return_counts=True)
        # print(dict(zip(unique, counts)))
        # print('Chords shape: {}  Melodies shape: {}'.format(chords.shape, melodies.shape))

        # Train test split
        self.__prepare_data_tt_splited(tt_split=tt_split, model_name=model_name)
        # print('Chords shape: {}  Melodies shape: {}'.format(chords.shape, melodies.shape))


        # Load / train model
        if model_name == 'basic_rnn':
            if os.path.exists("basic_rnn.h5"):
                model = load_model("basic_rnn.h5")
            else:
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_basic_rnn_model(input_dim=self.X_train.shape[1:])
                model = mb.train_model(model, epochs, loss="categorical_crossentropy")
                model.save("basic_rnn.h5")

        self.model = model

    def load_model(self, model_name):
        if model_name == 'basic_rnn':
            self.model = load_model('../note/basic_rnn.h5')
        else:
            print('No model name: {}'.format(model_name))

    def generate_notes_from_chord(self, chords, train_loss='softmax'):
        '''
        Generate notes from chords in test set, need to specify index.
        :param chords: chord piano roll - (128, x)
        :return: None. Write Midi out as melody.mid.
        '''
        # Prediction
        y = self.model.predict(np.expand_dims(np.transpose(chords, (1,0)), axis=0))

        # Handle probabilities according to training loss used
        if train_loss == 'softmax':
            b = np.zeros_like(y)
            b[np.arange(len(y)), np.arange(len(y[0])), y.argmax(2)] = 1

        # Matrix to piano roll
        y = np.transpose(np.squeeze(b), (1,0))
        y[y > 0] = 90
        chords[chords > 0] = 90
        # unique, counts = np.unique(chords, return_counts=True)
        # print(dict(zip(unique, counts)))

        # Write out as midi
        y_midi = piano_roll_to_pretty_midi(y, fs=12)
        y_midi.write('melody.mid')

    def __get_raw_data(self, src='nottingham', model_name='basic_rnn'):
        '''
        Get raw data depending on data source and model.
        :param src: Data source includes 'nottingham' and 'lakh'.
        :param model_name: Model includes 'basic_rnn'.
        :return:
        '''
        if src == 'nottingham':
            dp = DataPipeline()
            chords, melodies = dp.get_nottingham_piano_roll(is_small_set=False, is_shifted=False)

            # plt.imshow(chords[0])
            # plt.show()
            # plt.imshow(melodies[0])
            # plt.show()

            chords[chords > 0] = 1
            melodies[melodies > 0] = 1
            csparsity = 1.0 - np.count_nonzero(chords) / chords.size
            msparsity = 1.0 - np.count_nonzero(melodies) / melodies.size
            print(csparsity, msparsity)
            cshape, mshape = chords.shape, melodies.shape
            print(chords.shape, melodies.shape)

        else:
            pass

        chords, melodies = self.__process_raw_data(chords, melodies, model=model_name)
        return chords, melodies

    def __process_raw_data(self, chords, melodies, model='basic_rnn'):
        if model == 'basic_rnn':
            chords, melodies = np.transpose(chords, (0, 2, 1)), np.transpose(melodies, (0, 2, 1))

        return chords, melodies

    def __prepare_data_tt_splited(self, tt_split=0.9, src='nottingham', model_name='basic_rnn'):
        chords, melodies = self.__get_raw_data(src=src, model_name=model_name)
        split_ind = int(tt_split * len(chords))
        self.X_train, self.Y_train, self.X_test, self.Y_test = chords[:split_ind], \
                                                               melodies[:split_ind], \
                                                               chords[split_ind:], \
                                                               melodies[split_ind:]


if __name__ == "__main__":
    generator = ChordToNoteGenerator()
    generator.train_chord_to_melody_model(epochs=5000)
    #ind = 10

    #chords = np.transpose(generator.X_test[ind], (1,0))
    #from utils import piano_roll_to_pretty_midi

    #chords_temp = np.copy(chords)
    #chords_temp[chords_temp > 0] = 90
    #c_midi = piano_roll_to_pretty_midi(chords_temp, fs=12)
    #c_midi.write('chord.mid')

    #generator.generate_notes_from_chord(chords=chords)
    #from generator.song_generator import merge_melody_with_chords
    #merge_melody_with_chords('melody.mid', 'chord.mid', 'song.mid')

    #actual = np.transpose(generator.Y_test[ind], (1, 0))
    #actual[actual > 0] = 90
    #a_midi = piano_roll_to_pretty_midi(actual, fs=12)
    #a_midi.write('actual.mid')
    #merge_melody_with_chords('actual.mid', 'chord.mid', 'song-actual.mid')
