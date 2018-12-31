'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    A complete song generator.
'''

from chord.chord_generator import ChordGenerator, CHORD_SEQUENCE_FILE
from note.chord_to_note_generator import ChordToNoteGenerator
from dataset.data_pipeline import MAX_NUM_OF_BARS, FS
import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
from utils import piano_roll_to_pretty_midi
from mido import MidiFile


def generate_song(chords=None, bar_number=16):
    # 1. Generate chords
    chord_generator = ChordGenerator(CHORD_SEQUENCE_FILE)
    chords = chord_generator.generate_chords(chords, num_of_chords=bar_number)

    # 2. Convert chords to piano roll
    fp = chord_generator.chords_to_midi(chords)
    temp_chord_midi = pretty_midi.PrettyMIDI(fp)
    pr = temp_chord_midi.get_piano_roll(fs=12)

    # pad chord piano roll
    if pr.shape[-1] < MAX_NUM_OF_BARS * FS:
        to_pad = MAX_NUM_OF_BARS * FS - pr.shape[-1]
        pr = np.pad(pr, [(0, 0), (0, to_pad)], mode='constant', constant_values=0)
    # plt.imshow(pr)
    # plt.show()

    # save a copy of chords midi file
    pr_save = np.copy(pr)
    pr_save[pr_save > 0] = 90
    pr_midi = piano_roll_to_pretty_midi(pr_save, fs=12)
    pr_midi.write('chords.mid')

    # 3. Generate notes given chords
    chord_to_note_generator = ChordToNoteGenerator()
    chord_to_note_generator.load_model('basic_rnn')
    chord_to_note_generator.generate_notes_from_chord(pr)

    # 4. Merge melody with chords
    merge_melody_with_chords('melody.mid', 'chords.mid', 'song.mid')
    print('Song generation done.')


def merge_melody_with_chords(melody_file, chord_file, song_file):
    melody = MidiFile(melody_file)
    chord = MidiFile(chord_file)
    melody.tracks.append(chord.tracks[-1])
    melody.save(song_file)


if __name__ == "__main__":
    generate_song(chords=['D:maj', 'A:maj', 'B:min', 'F#:min'])