import csv
import os

from keras.utils import to_categorical
from music21 import *
from tqdm import tqdm

from chord.chord_generator import ChordGenerator
from utils import chord_index_to_piano_roll, piano_roll_to_pretty_midi
import numpy as np

MAX_LENGTH = 1200


def main():
    filepath = 'csv-leadsheet/csv_test/'
    filenames = os.listdir(filepath)
    all_chord_lst = []
    all_note_lst = []

    for i in tqdm(range(len(filenames))):
        file = filenames[i]
        if "From This Moment On written" in file:
            continue

        chord_lst = []
        note_lst = []
        with open(filepath + file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                if 'time' in row:   # title row
                    continue

                # process chord
                row[4] = row[4][:-1] if row[4][-1] == '0' else row[4]
                cg = ChordGenerator()
                chord = row[4] + ':' + row[5]
                try:
                    chord_value = cg.chord_to_id(chord)
                except KeyError:
                    # print(row)
                    continue

                # process note
                if row[6] == 'rest':
                    note_value = 0
                else:
                    row[6] = row[6][:-1] if row[6][-1] == '0' else row[6]
                    n = note.Note(row[6] + row[7])
                    note_value = int(n.pitch.ps)

                for _ in range(int(float(row[8]))):
                    chord_lst.append(chord_value)
                    if note_value >= 128:       # not sure why we have pitch values > 128 here
                        note_lst.append(0)
                    else:
                        note_lst.append(note_value)

        # print(len(chord_lst) == len(note_lst))

        # generate chords from notes
        if len(chord_lst) > MAX_LENGTH:
            chord_lst = chord_lst[:MAX_LENGTH]
            note_lst = note_lst[:MAX_LENGTH]
        while len(chord_lst) < MAX_LENGTH:
            chord_lst.append(0)
        while len(note_lst) < MAX_LENGTH:
            note_lst.append(0)

        result_chord_pr = chord_index_to_piano_roll(chord_lst)
        chord_midi = piano_roll_to_pretty_midi(result_chord_pr, fs=12)
        chord_midi.write('chords.mid')

        # get the melody
        test_melody = to_categorical(note_lst, num_classes=128)
        test_melody = np.transpose(np.squeeze(test_melody), (1, 0))
        test_melody[test_melody > 0] = 90
        melody_midi = piano_roll_to_pretty_midi(test_melody, fs=12)
        melody_midi.write('melody.mid')

        all_chord_lst.append(chord_lst)
        all_note_lst.append(note_lst)

    np.save('chords_test.npy', np.array(all_chord_lst))
    np.save('notes_test.npy', np.array(all_note_lst))


if __name__ == "__main__":
    main()