'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    A complete song generator.
'''
import os

from chord.chord_generator import ChordGenerator, CHORD_SEQUENCE_FILE, CHORD_SEQUENCE_FILE_SHIFTED
from note.chord_to_note_generator import ChordToNoteGenerator
from dataset.data_pipeline import MAX_NUM_OF_BARS, FS
import pretty_midi
import numpy as np
from utils import piano_roll_to_pretty_midi
from mido import MidiFile


def generate_song(chords=None, bar_number=16, melody_instrument=0, chord_instrument=0,
                  is_normalized=True):
    # 1. Generate chords
    if is_normalized:
        chord_generator = ChordGenerator(CHORD_SEQUENCE_FILE_SHIFTED)
    else:
        chord_generator = ChordGenerator(CHORD_SEQUENCE_FILE)
    chords = chord_generator.generate_chords(chords, num_of_chords=bar_number)

    # 2. Convert chords to piano roll
    fp = chord_generator.chords_to_midi(chords)
    temp_chord_midi = pretty_midi.PrettyMIDI(fp)
    pr = temp_chord_midi.get_piano_roll(fs=12)

    pr_length = pr.shape[-1]

    # pad chord piano roll
    if pr_length < MAX_NUM_OF_BARS * FS:
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
    if is_normalized:
        chord_to_note_generator.load_model('basic_rnn_normalized')
    else:
        chord_to_note_generator.load_model('basic_rnn_unnormalized')
    chord_to_note_generator.generate_notes_from_chord(pr)

    # 4. Truncate the melody to be of the chords' length
    temp_melody_midi = pretty_midi.PrettyMIDI('melody.mid')
    melody_pr = temp_melody_midi.get_piano_roll(fs=12)
    melody_pr = melody_pr[:, :pr_length]
    print(melody_pr.shape)
    melody_midi = piano_roll_to_pretty_midi(melody_pr, fs=12)
    melody_midi.write('melody.mid')

    # 5. Post processing - merging, changing instruments, etc.
    song_post_processing('melody.mid', 'chords.mid', 'song.mid', melody_instrument=melody_instrument,
                         chord_instrument=chord_instrument)
    print('Song generation done.')


def merge_melody_with_chords(melody_file, chord_file, song_file):
    melody = MidiFile(melody_file)
    chord = MidiFile(chord_file)
    melody.tracks.append(chord.tracks[-1])
    melody.save(song_file)


def change_midi_instrument(file_name, target_instrument):
    mid = MidiFile(file_name)
    for track in mid.tracks:
        for msg in track:
            if hasattr(msg, 'program'):  # instrument attribute
                msg.program = target_instrument
    mid.save(file_name)
    print('Instrument changed to id - {}'.format(target_instrument))


def song_post_processing(melody_file, chord_file, song_file, melody_instrument, chord_instrument):
    change_midi_instrument(melody_file, melody_instrument)
    change_midi_instrument(chord_file, chord_instrument)
    merge_melody_with_chords(melody_file, chord_file, song_file)

# Techno instruments - 81, 93
# Strings instruments - 40, 48 / 52, 48


if __name__ == "__main__":
    for i in range(10):
        generate_song(chords=['B:min', 'G:maj', 'D:maj', 'A:maj'], bar_number=16,
                  melody_instrument=81, chord_instrument=93)
        os.rename('melody.mid', 'melody-{}.mid'.format(i))
        os.rename('chords.mid', 'chords-{}.mid'.format(i))
        os.rename('song.mid', 'song-{}.mid'.format(i))