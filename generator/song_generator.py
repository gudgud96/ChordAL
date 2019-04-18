'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    A complete song generator.
'''
import os
import pickle
import random
from collections import Counter
import matplotlib.pyplot as plt

from keras.utils import to_categorical

from chord.chord_generator import ChordGenerator, CHORD_SEQUENCE_FILE, CHORD_SEQUENCE_FILE_SHIFTED
from note.chord_to_note_generator import ChordToNoteGenerator
# from note2chord.note_to_chord_generator import NoteToChordGenerator
from dataset.data_pipeline import MAX_NUM_OF_BARS, FS, DataPipeline
import pretty_midi
import numpy as np
from utils import piano_roll_to_pretty_midi, merge_melody_with_chords
from mido import MidiFile
from music21 import scale

MODEL_NAME = "bidem_preload"
MODELS_TO_NOTE = ["attention", "bidem", "bidem_preload", "bidem_preload", "bidem_regularized"]


def generate_chords(chords, bar_number):
    chord_generator = ChordGenerator(CHORD_SEQUENCE_FILE_SHIFTED)

    if not chords:
        chords = ['D:maj', 'A:maj', 'B:min', 'F#:min']
    chords = chord_generator.generate_chords(chords, num_of_chords=bar_number)
    chords = chord_tuning(chords)
    return chords


def chord_tuning(chords):
    tonic_chord = chords[0]     # assume first chord to be in tonic
    tonic_key, tonic_tonality = tonic_chord.split(":")
    sc = scale.MajorScale(tonic_key) if tonic_tonality == "maj" \
        else scale.MinorScale(tonic_key)
    sc = [p.name for p in sc.getPitches(tonic_key + '2', tonic_key + '3')]
    # tune the ending part of the chord sequence
    ending_key, ending_tonality = chords[-1].split(':')
    leading_key, leading_tonality = chords[-2].split(':')

    if tonic_tonality == "maj":
        relative_minor_key = sc[5]
        dominant_key = sc[4]
        if ending_key == tonic_key:
            pass               # no changes if ending on tonic
        elif ending_key == relative_minor_key:
            if leading_key != dominant_key:
                chords[-2] = dominant_key + ":maj"
        else:
            chords[-2] = dominant_key + ":maj"
            if random.uniform(0, 1) > 0.8:
                chords[-1] = tonic_key + ":maj"
            else:
                chords[-1] = relative_minor_key + ":min"

    else:
        relative_major_key = sc[2]
        major_dominant_key = sc[-2]
        minor_dominant_key = sc[4]

        if ending_key == tonic_key:
            pass
        elif ending_key == relative_major_key:
            if leading_key != major_dominant_key:
                chords[-2] = major_dominant_key + ":maj"
        else:

            if random.uniform(0, 1) > 0.8:
                chords[-2] = major_dominant_key + ":maj"
                chords[-1] = relative_major_key + ":maj"
            else:
                chords[-2] = minor_dominant_key + ":maj"
                chords[-1] = tonic_key + ":min"

    return chords


def convert_chords_to_piano_roll(chords):
    cg = ChordGenerator()
    fp = cg.chords_to_midi(chords)
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

    return pr_save, pr, pr_length


def generate_song(chords=None, bar_number=16, melody_instrument=0, chord_instrument=0,
                  style='piano', model_name='bidem'):
    # 1. Generate chords
    chords = generate_chords(chords, bar_number=bar_number)

    # 2. Convert chords to piano roll
    pr_save, pr, pr_length = convert_chords_to_piano_roll(chords)

    # 2.5 If model is bidirectional with embedding, we need to convert pr to indices first
    if model_name in MODELS_TO_NOTE:
        dp = DataPipeline()
        # print(pr_save.shape)
        chord_indices = dp.convert_chord_to_indices(pr_save)
        # print(chord_indices.shape)

    # 3. Generate notes given chords
    chord_to_note_generator = ChordToNoteGenerator()
    chord_to_note_generator.load_model(model_name)

    if MODEL_NAME in MODELS_TO_NOTE:
        chord_to_note_generator.generate_notes_from_chord(chord_indices, is_bidem=True)
    else:
        chord_to_note_generator.generate_notes_from_chord(pr, is_bidem=False)

    # 4. Truncate the melody to be of the chords' length
    temp_melody_midi = pretty_midi.PrettyMIDI('melody.mid')
    melody_pr = temp_melody_midi.get_piano_roll(fs=12)
    melody_pr = melody_pr[:, :pr_length]

    # 5. Melody tuning
    melody_pr = melody_tuning(melody_pr)
    melody_midi = piano_roll_to_pretty_midi(melody_pr, fs=12)
    melody_midi.write('melody.mid')

    os.remove('chords_to_midi.mid')

    # 6. Post processing - merging, changing instruments, etc.
    song_styling('melody.mid', 'chords.mid', 'song.mid', melody_instrument=melody_instrument,
                 chord_instrument=chord_instrument, style=style, chords=chords)


def melody_tuning(melody):
    if melody.shape[0] == 128:
        melody = np.argmax(melody, axis=0)      # normalize to melody indices

    def count_lengths(seq):
        lengths = []
        count = 1
        last_index = 0
        for i in range(1, len(seq)):
            if seq[i] != seq[last_index]:
                lengths.append(count)
                last_index = i
                count = 1
            else:
                count += 1
        lengths.append(count)
        return lengths

    # determine if most of the notes last for odd or even lengths
    lengths = [k % 2 for k in count_lengths(melody)]
    is_odd = max(lengths, key=lengths.count)

    # smoothing using sliding window
    window_size = 3
    for i in range(len(melody) - window_size):
        left, center, right = melody[i : i + window_size]

        if i == 0:  # tuning needed at the start of the melody
            if left != center and center == right:
                melody[i] = center

        if left == right and center != right:
            melody[i + 1] = left

        elif left != center and center != right:
            left_len, j = 1, i - 1
            while j >= 0 and melody[j] == melody[i]:
                left_len += 1
                j -= 1
            right_len, j = 1, i + 3
            while j < len(melody) and melody[j] == melody[i + 2]:
                right_len += 1
                j += 1

            if left_len % 2 == is_odd:
                if right_len % 2 != is_odd:
                    melody[i + 1] = left
                else:
                    melody[i + 1] = left if left_len < right_len else right
            else:
                if right_len % 2 == is_odd:
                    melody[i + 1] = right
                else:
                    melody[i + 1] = left if left_len < right_len else right

    # only stick to 1 note if the note is a chord note in the last bar
    last_bar = list(melody[-20:])
    sustaining_note = max(last_bar, key=list(last_bar).count)
    sustaining_index = last_bar.index(sustaining_note)
    for i in range(sustaining_index, len(last_bar)):
        last_bar[i] = sustaining_note
    j = 0
    for i in range(len(melody) - 20, len(melody)):
        melody[i] = last_bar[j]
        j += 1

    tuned_melody = np.transpose(to_categorical(melody, num_classes=128), (1,0))
    tuned_melody[tuned_melody > 0] = 90
    return tuned_melody


def melody_tuning_test():
    melody_pr = pretty_midi.PrettyMIDI('melody.mid').get_piano_roll(fs=12)
    plt.imsave("before-tuning.png", melody_pr)
    tuned_melody_pr = melody_tuning(melody_pr)
    plt.imsave("after-tuning.png", tuned_melody_pr)

    tuned_melody_pr[tuned_melody_pr > 0] = 90
    mid = piano_roll_to_pretty_midi(tuned_melody_pr, fs=12)
    mid.write("after_tuning.mid")


def __reduce_chord_array(chord_array):
    result = [chord_array[0]]
    cur = chord_array[0]
    for k in chord_array:
        if k != cur:
            result.append(k)
            cur = k
    if k != result[-1]:
        result.append(k)
    return np.array(result)


def __resize_note_array(notes):
    notes = np.array(notes)
    if len(notes.shape) == 1:
        notes = np.expand_dims(notes, axis=-1)
    return np.squeeze(np.resize(notes, (24,1)))


def change_midi_instrument(file_name, target_instrument):
    mid = MidiFile(file_name)
    for track in mid.tracks:
        for msg in track:
            if hasattr(msg, 'program'):  # instrument attribute
                msg.program = target_instrument
    mid.save(file_name)
    # print('Instrument changed to id - {}'.format(target_instrument))


def song_styling(melody_file, chord_file, song_file, melody_instrument=0, chord_instrument=0,
                         style='custom', chords=None):
    if style == 'piano':
        print('Chosen style: Piano')
        change_midi_instrument(melody_file, 0)
        change_midi_instrument(chord_file, 0)
        merge_melody_with_chords(melody_file, chord_file, song_file)

    elif style == 'techno':
        print('Chosen style: Techno')
        change_midi_instrument(melody_file, 94)
        change_midi_instrument(chord_file, 93)
        merge_melody_with_chords(melody_file, chord_file, song_file)
        add_drum_beats(song_file)
        add_bass(song_file, chords)

    elif style == 'strings':
        print('Chosen style: Strings')
        change_midi_instrument(melody_file, 40)
        change_midi_instrument(chord_file, 49)
        merge_melody_with_chords(melody_file, chord_file, song_file)

    elif style == 'organ':
        print('Chosen style: Organ')
        change_midi_instrument(melody_file, 48)
        change_midi_instrument(chord_file, 19)
        merge_melody_with_chords(melody_file, chord_file, song_file)

    elif style == 'church':
        print('Chosen style: Church')
        change_midi_instrument(melody_file, 49)
        change_midi_instrument(chord_file, 52)
        merge_melody_with_chords(melody_file, chord_file, song_file)

    elif style == 'brass':
        print('Chosen style: Brass')
        change_midi_instrument(melody_file, 60)
        change_midi_instrument(chord_file, 58)
        merge_melody_with_chords(melody_file, chord_file, song_file)

    elif style == 'flute':
        print('Chosen style: Flute')
        change_midi_instrument(melody_file, 73)
        change_midi_instrument(chord_file, 89)
        merge_melody_with_chords(melody_file, chord_file, song_file)

    else:
        print('Chosen style: Custom')
        change_midi_instrument(melody_file, melody_instrument)
        change_midi_instrument(chord_file, chord_instrument)
        merge_melody_with_chords(melody_file, chord_file, song_file)


def add_drum_beats(song_file, beat_style=1):
    # choose beat style
    print(os.getcwd())
    if beat_style == 1:
        drum_mid = MidiFile('../generator/Alone.mid')

        # scaling with a magical factor of 2.3
        track = drum_mid.tracks[4]
        for msg in track:
            if hasattr(msg, 'time'):
                if msg.time > 0:
                    msg.time = int(msg.time * 2.3)
    else:
        track = None
        pass

    # append to original song
    mid = MidiFile(song_file)
    mid.tracks.append(track)
    mid.save('song.mid')


def add_bass(song_file, chords, bass_instrument=34):
    mid = MidiFile(song_file)
    cg = ChordGenerator('')
    cg.chords_to_midi(chords, name='bass_line.mid', is_chord=False)

    temp_chord_midi = pretty_midi.PrettyMIDI('bass_line.mid')
    pr = temp_chord_midi.get_piano_roll(fs=48)
    pr_save = np.copy(pr)
    pr_save[pr_save > 0] = 90
    pr_midi = piano_roll_to_pretty_midi(pr_save, fs=12)
    pr_midi.write('bass_line.mid')
    change_midi_instrument('bass_line.mid', bass_instrument)

    bass_line = MidiFile('bass_line.mid')
    mid.tracks.append(bass_line.tracks[-1])
    mid.save(song_file)
    os.remove('bass_line.mid')


if __name__ == "__main__":
    melody_tuning_test()