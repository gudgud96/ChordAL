'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Chord to note generator.
'''
import pickle
import sys,os

sys.path.append('/'.join(os.getcwd().split('/')[:-1]))

from keras import backend as K
from utils import piano_roll_to_pretty_midi, chord_index_to_piano_roll, convert_chord_indices_to_embeddings
from dataset.data_pipeline import DataPipeline
from models.model_builder import ModelBuilder
import numpy as np


class ChordToNoteGenerator:
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None
        self.model_name = None

    def train_chord_to_melody_model(self, tt_split=0.9, epochs=100, model_name='basic_rnn'):
        '''
        Train model step - model takes in chord piano roll and outputs melody piano roll.
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
                model = mb.build_basic_rnn_model(input_dim=self.X_train.shape[1:])
                model.load_weights("basic_rnn.h5")
            else:
                mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)
                model = mb.build_attention_bidirectional_rnn_model(input_dim=self.X_train.shape[1:])
                model = mb.train_model(model, epochs, loss="categorical_crossentropy")
                model.save_weights("basic_rnn.h5")

        self.model = model

    def load_model(self, model_name, tt_split=0.9, is_fast_load=True):
        # clear session to avoid any errors
        K.clear_session()

        print("Chosen model: {}".format(model_name))

        if not is_fast_load:
            # Train test split
            if model_name == 'bidem' or model_name == 'attention' or model_name == "bidem_preload":
                self.__prepare_data_tt_splited(tt_split=tt_split, model_name=model_name, src='nottingham-embed')
                print('Chords shape: {}  Melodies shape: {}'.format(self.X_train.shape, self.Y_train.shape))
            else:
                self.__prepare_data_tt_splited(tt_split=tt_split, model_name=model_name, src='nottingham')
                print('Chords shape: {}  Melodies shape: {}'.format(self.X_train.shape, self.Y_train.shape))

        if is_fast_load:
            mb = ModelBuilder(None, None, None, None)
        else:
            mb = ModelBuilder(self.X_train, self.Y_train, self.X_test, self.Y_test)

        if model_name == 'basic_rnn_normalized':
            self.model = mb.build_basic_rnn_model(input_dim=(1200, 128))
            weights_path = '../note/active_models/basic_rnn_weights_500.h5'
            print('Loading ' + weights_path + '...')
            self.model.load_weights(weights_path)

        elif model_name == 'basic_rnn_unnormalized':
            self.model = mb.build_basic_rnn_model(input_dim=(1200, 128))
            weights_path = '../note/active_models/basic_rnn_weights_500_unnormalized.h5'
            print('Loading ' + weights_path + '...')
            self.model.load_weights(weights_path)

        elif model_name == 'bidem':
            self.model = mb.build_bidirectional_rnn_model(input_dim=(1200,))
            weights_path = '../note/active_models/bidem_weights_500.h5'
            print('Loading ' + weights_path + '...')
            self.model.load_weights(weights_path)

        elif model_name == 'bidem_regularized':
            self.model = mb.build_bidirectional_rnn_model_no_embeddings(input_dim=(1200,1))
            weights_path = '../note/active_models/bidirectional_regularized_500.h5'
            print('Loading ' + weights_path + '...')
            self.model.load_weights(weights_path)

        elif model_name == 'attention':
            self.model = mb.build_attention_bidirectional_rnn_model(input_dim=(1200,))
            weights_path = '../note/active_models/attention_weights_1000.h5'
            print('Loading ' + weights_path + '...')
            self.model.load_weights(weights_path)

        elif model_name == 'bidem_preload':
            self.model = mb.build_bidirectional_rnn_model_no_embeddings(input_dim=(1200, 32))
            weights_path = '../note/active_models/bidirectional_embedding_preload.h5'
            print('Loading ' + weights_path + '...')
            self.model.load_weights(weights_path)

        else:
            print('No model name: {}'.format(model_name))
            return

        self.model_name = model_name

    def generate_notes_from_chord(self, chords, train_loss='softmax', is_bidem=True, is_return=False):
        '''
        Generate notes from chords in test set, need to specify index.
        :param chords: chord piano roll - (128, x)
        :return: None. Write Midi out as melody.mid.
        '''
        # Preprocessing
        if self.model_name == "bidem_preload":
            chords = convert_chord_indices_to_embeddings(chords)
        elif self.model_name == "bidem_regularized":
            chords = np.expand_dims(chords, axis=-1)

        # Prediction
        if is_bidem:
            y = self.model.predict(np.expand_dims(chords, axis=0))
        else:
            y = self.model.predict(np.expand_dims(np.transpose(chords, (1,0)), axis=0))

        # Handle probabilities according to training loss used
        if train_loss == 'softmax':
            b = np.zeros_like(y)
            b[np.arange(len(y)), np.arange(len(y[0])), y.argmax(2)] = 1

        # Matrix to piano roll
        y = np.transpose(np.squeeze(b), (1,0))
        y[y > 0] = 90
        chords[chords > 0] = 90

        # Write out as midi
        y_midi = piano_roll_to_pretty_midi(y, fs=12)
        y_midi.write('melody.mid')

        if is_return:
            return y

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
            chords, melodies = self.__process_raw_data(chords, melodies, model=model_name)
            print(chords.shape, melodies.shape)

        elif src == 'nottingham-embed':
            dp = DataPipeline()
            chords, melodies = dp.get_nottingham_embed(is_small_set=True)
            melodies[melodies > 0] = 1
            cshape, mshape = chords.shape, melodies.shape
            print(chords.shape, melodies.shape)

        return chords, melodies

    def __process_raw_data(self, chords, melodies, model='basic_rnn'):
        if model == 'basic_rnn':
            chords, melodies = np.transpose(chords, (0, 2, 1)), np.transpose(melodies, (0, 2, 1))
        elif model == 'basic_rnn_embed':
            melodies = np.transpose(melodies, (0, 2, 1))

        return chords, melodies

    def __prepare_data_tt_splited(self, tt_split=0.9, src='nottingham', model_name='basic_rnn'):
        chords, melodies = self.__get_raw_data(src=src, model_name=model_name)
        if src == 'nottingham-embed':
            chords, melodies = self.__process_raw_data(chords, melodies, model='basic_rnn_embed')
        else:
            chords, melodies = self.__process_raw_data(chords, melodies, model='basic_rnn')
        split_ind = int(tt_split * len(chords))
        self.X_train, self.Y_train, self.X_test, self.Y_test = chords[:split_ind], \
                                                               melodies[:split_ind], \
                                                               chords[split_ind:], \
                                                               melodies[split_ind:]


def evaluate_preload_embeddings():
    model_name = "bidem"
    generator = ChordToNoteGenerator()
    generator.load_model(model_name=model_name, is_fast_load=False)

    ind = 6
    # chords = np.transpose(generator.X_test[ind], (1,0))
    chords = generator.X_test[ind]
    print(chords.shape)
    from utils import piano_roll_to_pretty_midi

    chords_temp = np.copy(chords)
    chord_midi = piano_roll_to_pretty_midi(chord_index_to_piano_roll(chords_temp), fs=12)
    chord_midi.write('chord.mid')

    # this is only for embedding preload
    if model_name == "bidem_preload":
        chords_embeddings = []
        infile = open('../dataset/chord_embeddings_dict.pickle', 'rb')
        embedding_dict = pickle.load(infile)
        for chord in chords:
            if chord == 0:
                chords_embeddings.append(np.zeros((32,)))
            else:
                chords_embeddings.append(embedding_dict[chord])
        chords = np.array(chords_embeddings)

    chords = np.expand_dims(chords, axis=-1)

    generator.generate_notes_from_chord(chords=chords)
    from generator.song_generator import merge_melody_with_chords
    merge_melody_with_chords('melody.mid', 'chord.mid', 'song.mid')

    actual = np.transpose(generator.Y_test[ind], (1, 0))
    actual[actual > 0] = 90
    a_midi = piano_roll_to_pretty_midi(actual, fs=12)
    a_midi.write('actual.mid')
    merge_melody_with_chords('actual.mid', 'chord.mid', 'song-actual.mid')


def beam_search():
    cg = ChordToNoteGenerator()
    cg.train_chord_to_melody_model(model_name="seq2seq", epochs=3)


if __name__ == "__main__":
    # evaluate_preload_embeddings()
    beam_search()