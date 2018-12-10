'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Finale module - network model on lead sheet generation given notes and chords
            as training data.

Improvements needed:
(/) - With autoencoder used, think how could the generation be done. (Use VAE)
( ) - We have the first result! But it is exceptionally awful. Optimize it!!
( ) - Seems like we cannot have polyphonic notes in our data.
( ) - For VAE, how to generate for not copying the training data??
( ) - Data used is not enough. When code structure is more flexible, should incorporate more data.
'''
import os

from keras import Sequential, Input, Model
from keras.layers import Dense, RepeatVector, LSTM, K
from keras.models import load_model
from keras.utils import to_categorical
from tqdm import tqdm

from rhythm.rhythm_generator import RhythmGenerator
from rhythm.rhythm_extractor import RhythmExtractor
from chord.generator.chord_generator import ChordGenerator, CHORD_DICT
from chord.extractor.chord_extractor_generic import ChordExtractor
from models.model_builder import ModelBuilder
from dataset.data_pipeline import DataPipeline
from music21 import *
import numpy as np
import matplotlib.pyplot as plt

MAX_BEAT_LENGTH = 1000
MAX_CHORD_LENGTH = 200
MAX_NOTE_LENGTH = 500
EPOCH_NUM = 1000
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

    def build_model(self, builder, input_dim):
        '''
        Build a model using ModelBuilder.
        :param X_train
        :param X_test
        :return: model
        '''
        # model = builder.build_stacked_autoencoder(input_dim, [512,256,128,256,512])
        # model = builder.train_model(model, EPOCH_NUM)
        # model.save('note_model.h5')
        # return model

        vae, encoder, decoder = builder.build_and_train_vae(input_dim, 256, 128, EPOCH_NUM)
        vae.save_weights('note_model_vae.h5')
        encoder.save_weights('note_model_encoder.h5')
        decoder.save_weights('note_model_decoder.h5')

        return vae, encoder, decoder

    def train_model(self):

        # load score data
        # scores = converter.parse(FILENAME)
        # if not os.path.exists('ashover-data.npy'):
        #     score_vectors = []
        #     note_vectors = []
        #     for score in tqdm(scores):
        #         score_vector, note_vector = self.get_train_test_data(score, is_stacked=True)
        #         score_vectors.append(score_vector)
        #         note_vectors.append(note_vector)
        #     score_vectors = np.array(score_vectors)
        #     note_vectors = np.array(note_vectors)
        #     print('note vector shape', note_vectors.shape)
        #     print(note_vectors)
        #     np.save('ashover-data.npy', score_vectors)
        # else:
        #     score_vectors = np.load('ashover-data.npy')
        dp = DataPipeline()
        score_vectors = dp.get_nottingham_data(is_stacked=True)
        score_vectors = score_vectors.reshape(score_vectors.shape[0], score_vectors.shape[1])

        # load models
        builder = ModelBuilder(score_vectors, score_vectors, score_vectors, score_vectors)
        if not os.path.exists('note_model_vae.h5'):
            vae, encoder, generator = self.build_model(builder, score_vectors.shape[-1])
        else:
            vae, encoder, generator, _, _ = builder.build_vae_model(score_vectors.shape[-1], 256, 128)

        # Predict any score
        index = 1
        target = score_vectors[index].reshape(1, score_vectors[index].shape[0])

        # two step predict
        z_mean, z_log_var, z = encoder.predict(target)
        print("min max", np.amin(z), np.amax(z), z.shape)
        predict_vector_1 = generator.predict(z)
        predict_vector_1 = np.rint(predict_vector_1[0])
        # self.__evaluate_predict_vector(predict_vector_1, target)

        # one step predict
        predict_vector_2 = vae.predict(target)
        predict_vector_2 = np.rint(predict_vector_2[0])
        predict_vector_2 = self.predict_vector_check(predict_vector_2)
        self.__evaluate_predict_vector(predict_vector_2, target)

        # scores[index].show()
        self.convert_vector_to_score(predict_vector_2)

        # generate from noise
        batch, dim = z_mean.shape
        epsilon = np.random.normal(0, 1, (batch,dim))
        seed = z_mean + np.exp(0.5 * z_log_var) * epsilon

        seed = np.random.normal(0, 1, (batch,dim))
        print("min max", np.amin(seed), np.amax(seed), seed.shape)

        generated_vector = generator.predict(seed)
        generated_vector = np.rint(generated_vector[0])
        generated_vector = self.predict_vector_check(generated_vector)

        # scores[index].show()
        self.convert_vector_to_score(generated_vector)

    def predict_vector_check(self, predict_vector):
        beat_anomaly = [i for i in predict_vector[:1000] if i < 0 or i > 2]
        chord_anomaly = [i for i in predict_vector[1000:1200] if i < 0 or i > 25]
        note_anomaly = [i for i in predict_vector[1200:1700] if i < 0 or i > 12]
        print("Beat anomaly: {} Chord anomaly: {} Note anomaly: {}".format(len(beat_anomaly) / 1000,
                                                                           len(chord_anomaly) / 200,
                                                                           len(note_anomaly) / 500))

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

        # print(beat_vector)

        # remove trailing zeros used for model training
        # problem: among the zeros there are some ones. problem comes from using m.s.e.
        cur_beat, cur_note = 0, 0
        while cur_beat == 0:
            cur_beat = beat_vector.pop()
        while cur_note == 0:
            cur_note = note_vector.pop()

        rhythm_generator = RhythmGenerator('')
        beat_pattern = rhythm_generator.decode_beats_from_rhythm_id(np.array(beat_vector))
        rhythm_generator.show_beats_as_notes(beat_pattern, pitch_pattern=note_vector)

    def __convert_tensor_to_one_hot(self, predict_vector):
        one_hot_predict_vector = np.zeros_like(predict_vector)

        # TODO: optimize this, don't use for loop
        for i in range(len(one_hot_predict_vector)):
            # TODO: use random sampling, seems like the 1 note problem recurs again
            one_hot_predict_vector[i][np.argmax(predict_vector[i])] = 1

        return one_hot_predict_vector

    def __evaluate_predict_vector(self, predict_vector, target):
        '''
        Evaluate the accuracy of 3 aspects of predict vector
        :param predict_vector:
        :param target:
        :return:
        '''
        def evaluate(predict, target, string):
            acc_arr = (predict == target).astype(int)
            print('{}: {}'.format(string, np.count_nonzero(acc_arr) / acc_arr.shape[-1]))

        print('==========   Evaluation   ============')
        evaluate(predict_vector, target, "Overall test acc")
        evaluate(predict_vector[:1000], target[0,:1000], "Beat test acc")
        evaluate(predict_vector[1000:1200], target[0,1000:1200], "Chord test acc")
        evaluate(predict_vector[1200:], target[0,1200:], "Note test acc")
        # print(target[0,:1000])
        print('======================================')


def main():
    note_generator = NoteGenerator()
    note_generator.train_model()


if __name__ == "__main__":
    main()

