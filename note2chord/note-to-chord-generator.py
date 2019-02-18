'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Note to chord generator.
'''
import numpy as np
import os

from keras.utils import to_categorical

from dataset.data_pipeline import DataPipeline
from models.model_builder import ModelBuilder


class NoteToChordGenerator:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None

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
            if os.path.exists("basic_rnn.h5"):
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_basic_rnn_model(input_dim=self.X_train.shape[1:],
                                                 output_dim=self.Y_train.shape[-1])
                model.load_weights("basic_rnn.h5")
            else:
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_basic_rnn_model(input_dim=self.X_train.shape[1:],
                                                 output_dim=self.Y_train.shape[-1])
                model = mb.train_model(model, epochs, loss="categorical_crossentropy")
                model.save_weights("basic_rnn.h5")

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
    generator.train_melody_to_chord_model(epochs=5, model_name='basic_rnn')