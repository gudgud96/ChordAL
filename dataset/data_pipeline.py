'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Data pipeline module to meet data requirements for training models.

Improvements needed:
(/) - Include Nottingham Dataset pipeline.
( ) - Include Lakh Dataaset pipeline.
'''
import os
import time

import pretty_midi
from keras.utils import to_categorical
from tqdm import tqdm

from chord.chord_extractor_generic import ChordExtractor
from chord.chord_generator import ChordGenerator, CHORD_DICT
# from rhythm.rhythm_extractor import RhythmExtractor
# from rhythm.rhythm_generator import RhythmGenerator
from music21 import *
import numpy as np

MAX_BEAT_LENGTH = 1000
MAX_CHORD_LENGTH = 200
MAX_NOTE_LENGTH = 500
MAX_NUM_OF_BARS = 100
FS = 12

class DataPipeline:
    def __init__(self):
        pass

    def get_nottingham_data(self, is_reset=False, is_stacked=False):
        '''
        Public method to get Nottingham Dataset.
        :param is_reset: Whether to reset the numpy data
        :return:    all scores of shape (1034, 1200, 1)
                    all notes of shape (1034, 500, 13)      # categorical
        '''
        FILENAME = '../dataset/Nottingham-Data/'
        files = os.listdir(FILENAME)
        
        # reinitialize if needed
        if is_reset or len(files) == 0:
            self.__save_nottingham_data()

        notes_vector = []
        scores_vector = []
        for file in files:
            temp = np.load(FILENAME + file)
            if "notes" in file:
                notes_vector.append(temp)
            elif "scores" in file:
                scores_vector.append(temp)

        notes_vector = np.array(notes_vector)
        scores_vector = np.array(scores_vector)

        all_notes = notes_vector[0]
        for i in range(1, len(notes_vector)):
            all_notes = np.append(all_notes, notes_vector[i], axis=0)
        all_scores = scores_vector[0]
        for i in range(1, len(scores_vector)):
            all_scores = np.append(all_scores, scores_vector[i], axis=0)

        if is_stacked:
            all_notes = np.argmax(all_notes, axis=2).reshape(all_notes.shape[0], all_notes.shape[1], 1)
            return np.append(all_scores, all_notes, axis=1)
        else:
            return all_scores, all_notes

    def __save_nottingham_data(self):
        '''
        Save Nottingham Dataset into numpy format. From abc version.
        '''
        FILENAME = 'Nottingham/'
        SAVENAME = 'Nottingham-Data/'
        files = os.listdir(FILENAME)
        print("A total of {} files for Nottingham dataset. Loading...".format(len(files)))

        for file in files:
            fname = FILENAME + file
            print(fname)
            scores = converter.parse(fname)
            score_vectors = []
            note_vectors = []
            for score in tqdm(scores):
                score_vector, note_vector = self.__get_nottingham_score(fname, score, is_stacked=False)
                score_vectors.append(score_vector)
                note_vectors.append(note_vector)
            score_vectors = np.array(score_vectors)
            note_vectors = np.array(note_vectors)
            print('score vector shape', score_vectors.shape)
            print('note vector shape', note_vectors.shape)
            np.save(SAVENAME + file + '-scores.npy', score_vectors)
            np.save(SAVENAME + file + '-notes.npy', note_vectors)

    def __get_nottingham_score(self, fname, score, is_stacked=False):
        '''
        Return Nottingham score as numpy array.
        :param fname: filename of the score
        :param score: score object
        :param is_stacked: a boolean to determine if the result needs to be stacked into (1700,1).
        :return:
            If not is_stacked (by default), return scores and notes vector of shape (1200,1), (500,13)
        '''
        rhythm_extractor = RhythmExtractor(fname)
        beat_pattern = rhythm_extractor.extract_rhythm_to_text(score)
        chord_extractor = ChordExtractor()
        chord_pattern = chord_extractor.extract_chords(score)

        # print(beat_pattern)
        # print(chord_pattern)

        chord_generator = ChordGenerator(fname)
        one_hot_chord_pattern = chord_generator.one_hot_encode_chords(chord_pattern)
        chord_pattern = np.argmax(one_hot_chord_pattern, axis=1).reshape(-1, 1)
        if len(chord_pattern) > MAX_CHORD_LENGTH:
            chord_pattern = chord_pattern[:MAX_CHORD_LENGTH]

        rhythm_generator = RhythmGenerator(fname)
        one_hot_beat_pattern = rhythm_generator.one_hot_encode_beats(beat_pattern)
        beat_pattern = np.argmax(one_hot_beat_pattern, axis=1).reshape(-1, 1)
        if len(beat_pattern) > MAX_BEAT_LENGTH:
            beat_pattern = beat_pattern[:MAX_BEAT_LENGTH]

        # Prepare input data - #beats + zeros + #chords + zeros
        stacked_input_pattern = np.append(beat_pattern, np.zeros(MAX_BEAT_LENGTH - len(beat_pattern)))
        stacked_input_pattern = np.append(stacked_input_pattern, chord_pattern)
        stacked_input_pattern = np.append(stacked_input_pattern, np.zeros(MAX_CHORD_LENGTH
                                                                          - len(chord_pattern))).reshape(-1, 1)

        note_pattern = chord_extractor.extract_measures_from_score(score)
        flatten_note_pattern = [note.replace('-', 'b') for measure in note_pattern
                                for note in measure]

        # chord is represented by note name, so we can use this to encode notes too
        # prepare output data - #notes + zeros
        encoded_note_pattern = np.array([CHORD_DICT[note] for note in flatten_note_pattern])
        if len(encoded_note_pattern) > MAX_NOTE_LENGTH:
            encoded_note_pattern = encoded_note_pattern[:MAX_NOTE_LENGTH]
        encoded_note_pattern = np.append(encoded_note_pattern, np.zeros(MAX_NOTE_LENGTH -
                                                                        len(encoded_note_pattern)))
        one_hot_note_pattern = to_categorical(encoded_note_pattern, num_classes=13)

        if is_stacked:
            return np.append(stacked_input_pattern, encoded_note_pattern)
        else:
            return stacked_input_pattern, one_hot_note_pattern

    def get_nottingham_piano_roll(self, is_reset=False, is_small_set=True, is_shifted=True):
        '''
        Public method to get Nottingham Dataset piano roll.
        :param is_reset: to reset all saved data or not
        :param is_small_set: whether to get the small set
        :return: chords tensors and melodies tensors of shape (x, 100, 128, 12)
        '''
        if is_reset:
            self.__save_nottingham_piano_roll(0, 200, 1, is_shifted=is_shifted)
            self.__save_nottingham_piano_roll(200, 400, 2, is_shifted=is_shifted)
            self.__save_nottingham_piano_roll(400, 600, 3, is_shifted=is_shifted)
            self.__save_nottingham_piano_roll(600, 800, 4, is_shifted=is_shifted)
            self.__save_nottingham_piano_roll(800, 1021, 5, is_shifted=is_shifted)
            self.__merge_nottingham_piano_roll()

        print("Loading Nottingham Piano Roll...")
        shifted_fname = "Nottingham-Piano-shifted"
        non_shifted_fname = "Nottingham-Piano"
        t1 = time.time()
        if is_small_set:
            if is_shifted:
                print('Loading from {}'.format("../dataset/" + shifted_fname))
                chords_1 = np.load("../dataset/" + shifted_fname + "/chords-1.npy")
                melodies_1 = np.load("../dataset/" + shifted_fname + "/melodies-1.npy")
            else:
                print('Loading from {}'.format("../dataset/" + non_shifted_fname))
                chords_1 = np.load("../dataset/" + non_shifted_fname + "/chords-1.npy")
                melodies_1 = np.load("../dataset/" + non_shifted_fname + "/melodies-1.npy")
        else:
            if is_shifted:
                print('Loading from {}'.format("../dataset/" + shifted_fname))
                chords_1 = np.load("../dataset/" + shifted_fname + "/chords.npy")
                melodies_1 = np.load("../dataset/" + shifted_fname + "/melodies.npy")
            else:
                print('Loading from {}'.format("../dataset/" + non_shifted_fname))
                chords_1 = np.load("../dataset/" + non_shifted_fname + "/chords.npy")
                melodies_1 = np.load("../dataset/" + non_shifted_fname + "/melodies.npy")

        # reshape
        # chords_1 = chords_1.reshape(200, 1200, 128)
        # melodies_1 = melodies_1.reshape(200, 1200, 128)

        print(chords_1.shape)
        print(melodies_1.shape)
        print("Loading takes {} seconds.".format(time.time() - t1))
        return chords_1, melodies_1

    def get_nottingham_embed(self, is_small_set=True):
        '''
        Public method to get Nottingham Dataset embed.
        :param is_reset: to reset all saved data or not
        :param is_small_set: whether to get the small set
        :return: chords tensors and melodies tensors of shape (x, 100, 128, 12)
        '''
        print("Loading Nottingham Embed...")
        shifted_fname = "Nottingham-Piano-shifted"
        t1 = time.time()
        if is_small_set:
            print('Loading from {}'.format("../dataset/" + shifted_fname))
            chords_1 = np.load("../dataset/" + shifted_fname + "/chords-1-embedded.npy")
            melodies_1 = np.load("../dataset/" + shifted_fname + "/melodies-1.npy")

        else:
            print('Loading from {}'.format("../dataset/" + shifted_fname))
            chords_1 = np.load("../dataset/" + shifted_fname + "/chords-embedded.npy")
            melodies_1 = np.load("../dataset/" + shifted_fname + "/melodies.npy")

        # reshape
        # chords_1 = chords_1.reshape(200, 1200, 128)
        # melodies_1 = melodies_1.reshape(200, 1200, 128)

        print(chords_1.shape)
        print(melodies_1.shape)
        print("Loading takes {} seconds.".format(time.time() - t1))
        return chords_1, melodies_1

    def get_nottingham_halved(self):
        print("Loading Nottingham Halved...")
        fname = "csv-nottingham"
        t1 = time.time()
        print('Loading from {}'.format("../dataset/" + fname))
        chords_1 = np.load("../dataset/" + fname + "/chords_halved_nottingham.npy")
        melodies_1 = np.load("../dataset/" + fname + "/melodies_halved_nottingham.npy")

        print(chords_1.shape)
        print(melodies_1.shape)
        print("Loading takes {} seconds.".format(time.time() - t1))
        return chords_1, melodies_1

    def get_csv_nottingham_cleaned(self):
        print('Loading csv-nottingham cleaned...')
        fname = "csv-nottingham"
        t1 = time.time()
        print('Loading from {}'.format("../dataset/" + fname))
        chords_1 = np.load("../dataset/" + fname + "/chords_merged_cleaned.npy")
        melodies_1 = np.load("../dataset/" + fname + "/melodies_merged_cleaned.npy")

        print(chords_1.shape)
        print(melodies_1.shape)
        print("Loading takes {} seconds.".format(time.time() - t1))
        return chords_1, melodies_1

    def __save_nottingham_piano_roll(self, start, end, i, FS=12, is_shifted=True):
        if not is_shifted:
            self.__save_nottingham_piano_roll_impl(start, end, i, '../dataset/Nottingham-midi/melody/',
                                              '../dataset/Nottingham-midi/chords/')
        else:
            self.__save_nottingham_piano_roll_impl(start, end, i, '../dataset/Nottingham-shifted-midi/melody/',
                                              '../dataset/Nottingham-shifted-midi/chords/')

    def __save_nottingham_piano_roll_impl(self, start, end, i, melody_fname, chord_fname, FS=12):
        MELODY_FNAME = melody_fname
        CHORD_FNAME = chord_fname

        filelist = os.listdir(CHORD_FNAME)
        melody_prs = []
        chord_prs = []
        # i = 0
        for file in tqdm(filelist[start:end]):
            melody_midi = pretty_midi.PrettyMIDI(MELODY_FNAME + file)
            chord_midi = pretty_midi.PrettyMIDI(CHORD_FNAME + file)
            melody_pr = melody_midi.get_piano_roll(fs=FS)
            # print(melody_pr.shape)
            if melody_pr.shape[-1] < MAX_NUM_OF_BARS * FS:
                to_pad = MAX_NUM_OF_BARS * FS - melody_pr.shape[-1]
                melody_pr = np.pad(melody_pr, [(0,0), (0,to_pad)], mode='constant', constant_values=0)
            else:
                melody_pr = melody_pr[:, :MAX_NUM_OF_BARS * FS]

            chord_pr = chord_midi.get_piano_roll(fs=FS)
            if chord_pr.shape[-1] < MAX_NUM_OF_BARS * FS:
                to_pad = MAX_NUM_OF_BARS  * FS - chord_pr.shape[-1]
                chord_pr = np.pad(chord_pr, [(0,0), (0,to_pad)], mode='constant', constant_values=0)
            else:
                chord_pr = chord_pr[:, :MAX_NUM_OF_BARS * FS]

            melody_pr = melody_pr.reshape(128, MAX_NUM_OF_BARS * FS)
            chord_pr = chord_pr.reshape(128, MAX_NUM_OF_BARS * FS)

            melody_prs.append(melody_pr)
            chord_prs.append(chord_pr)

        np.save('melodies-{}.npy'.format(i), melody_prs)
        np.save('chords-{}.npy'.format(i), chord_prs)

    def convert_nottingham_chords_to_indices(self):
        chords = np.load('../dataset/Nottingham-Piano-shifted/chords.npy')
        new_chords = []
        for i in tqdm(range(len(chords))):
            chord = self.convert_chord_to_indices(chords[i])
            new_chords.append(chord)

        new_chords = np.array(new_chords)
        np.save('chords-embedded.npy', new_chords)

    def convert_chord_to_indices(self, chord):
        '''
        Takes in a piano roll of a song and convert to indices.
        :param chord: Piano roll of chords
        :return:
        '''
        chord = np.transpose(chord, (1, 0))
        chord_index_dict = {
            (36, 40): "C:maj", (36, 39): "C:min",
            (37, 41): "C#:maj", (37, 40): "C#:min",
            (38, 42): "D:maj", (38, 41): "D:min",
            (39, 43): "D#:maj", (39, 42): "D#:min",
            (40, 44): "E:maj", (40, 43): "E:min",
            (41, 45): "F:maj", (41, 44): "F:min",
            (42, 46): "F#:maj", (42, 45): "F#:min",
            (43, 47): "G:maj", (43, 46): "G:min",
            (44, 48): "G#:maj", (44, 47): "G#:min",
            (45, 49): "A:maj", (45, 48): "A:min",
            (46, 50): "A#:maj", (46, 49): "A#:min",
            (47, 51): "B:maj", (47, 50): "B:min",
            (48, 52): "C:maj", (48, 51): "C:min",
            (49, 53): "C#:maj", (49, 52): "C#:min",
            (50, 54): "D:maj", (50, 53): "D:min",
            (51, 55): "D#:maj", (51, 54): "D#:min",
            (52, 56): "E:maj", (52, 55): "E:min",
            (53, 57): "F:maj", (53, 56): "F:min",
            (54, 58): "F#:maj", (54, 57): "F#:min",
            (55, 59): "G:maj", (55, 58): "G:min",
            (56, 60): "G#:maj", (56, 59): "G#:min",
            (57, 61): "A:maj", (57, 60): "A:min",
            (57, 62): "A#:maj", (58, 61): "A#:min",
            (59, 63): "B:maj", (59, 62): "B:min",
            (0, 0, 0): "-"
        }
        new_chord = []
        for i in range(len(chord)):
            chord_t = chord[i]
            indices = np.where(chord_t > 0)[0]
            cg = ChordGenerator()
            if indices.size == 0:
                value = cg.chord_to_id(chord_index_dict[tuple(np.array([0, 0, 0]))])
                new_chord.append(value)
            else:
                value = cg.chord_to_id(chord_index_dict[tuple(indices[:2])])
                new_chord.append(value)

        return np.array(new_chord)

    def __merge_nottingham_piano_roll(self):
        chords = np.load('chords-1.npy')
        melodies = np.load('melodies-1.npy')
        for i in range(2, 6):
            temp_c = np.load('chords-{}.npy'.format(i))
            temp_m = np.load('melodies-{}.npy'.format(i))
            chords = np.concatenate((chords, temp_c), axis=0)
            melodies = np.concatenate((melodies, temp_m), axis=0)

        np.save('chords.npy', chords)
        np.save('melodies.npy', melodies)

    def get_lakh_data(self):
        CHORD_FNAME = '../dataset/Nottingham-midi/chords/'
        filelist = os.listdir(CHORD_FNAME)
        print(len(filelist))


if __name__ == "__main__":
    a = DataPipeline()
    # a.get_nottingham_piano_roll(is_small_set=False, is_reset=True)
    a.convert_nottingham_chords_to_indices()