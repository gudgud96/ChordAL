'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    A complete song generator.
'''
import os

from chord.chord_generator import ChordGenerator, CHORD_SEQUENCE_FILE, CHORD_SEQUENCE_FILE_SHIFTED
from note.chord_to_note_generator import ChordToNoteGenerator
from dataset.data_pipeline import MAX_NUM_OF_BARS, FS, DataPipeline
import pretty_midi
import numpy as np
from utils import piano_roll_to_pretty_midi
from mido import MidiFile

IS_BIDEM = True

def generate_song(chords=None, bar_number=16, melody_instrument=0, chord_instrument=0, style='piano',
                  is_normalized=True):
    # 1. Generate chords
    if is_normalized:
        chord_generator = ChordGenerator(CHORD_SEQUENCE_FILE_SHIFTED)
    else:
        chord_generator = ChordGenerator(CHORD_SEQUENCE_FILE)
    if not chords:
        chords = ['D:maj', 'A:maj', 'B:min', 'F#:min']
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

    # 2.5 If model is bidirectional with embedding, we need to convert pr to indices first
    if IS_BIDEM:
        dp = DataPipeline()
        print(pr_save.shape)
        chord_indices = dp.convert_chord_to_indices(pr_save)
        print(chord_indices.shape)

    # 3. Generate notes given chords
    chord_to_note_generator = ChordToNoteGenerator()
    if is_normalized:
        if IS_BIDEM:
            chord_to_note_generator.load_model('bidem')
        else:
            chord_to_note_generator.load_model('basic_rnn_normalized')
    else:
        chord_to_note_generator.load_model('basic_rnn_unnormalized')

    if IS_BIDEM:
        chord_to_note_generator.generate_notes_from_chord(chord_indices, is_bidem=IS_BIDEM)
    else:
        chord_to_note_generator.generate_notes_from_chord(pr, is_bidem=IS_BIDEM)

    # 4. Truncate the melody to be of the chords' length
    temp_melody_midi = pretty_midi.PrettyMIDI('melody.mid')
    melody_pr = temp_melody_midi.get_piano_roll(fs=12)
    melody_pr = melody_pr[:, :pr_length]
    print(melody_pr.shape)
    melody_midi = piano_roll_to_pretty_midi(melody_pr, fs=12)
    melody_midi.write('melody.mid')

    os.remove('chords_to_midi.mid')

    # 5. Post processing - merging, changing instruments, etc.
    song_styling('melody.mid', 'chords.mid', 'song.mid', melody_instrument=melody_instrument,
                         chord_instrument=chord_instrument, style=style, chords=chords)
    print('Song generation done.')


def merge_melody_with_chords(melody_file, chord_file, song_file):
    melody = MidiFile(melody_file)
    chord = MidiFile(chord_file)
    melody.tracks.append(chord.tracks[-1])

    # change each track to different channel
    for i in range(len(melody.tracks)):
        track = melody.tracks[i]
        for msg in track:
            if hasattr(msg, 'channel'):
                msg.channel = i

    melody.save(song_file)


def change_midi_instrument(file_name, target_instrument):
    mid = MidiFile(file_name)
    for track in mid.tracks:
        for msg in track:
            if hasattr(msg, 'program'):  # instrument attribute
                msg.program = target_instrument
    mid.save(file_name)
    print('Instrument changed to id - {}'.format(target_instrument))


def song_styling(melody_file, chord_file, song_file, melody_instrument=0, chord_instrument=0,
                         style='custom', chords=None):
    if style == 'piano':
        print('Chosen style: Piano')
        change_midi_instrument(melody_file, 0)
        change_midi_instrument(chord_file, 0)
        merge_melody_with_chords(melody_file, chord_file, song_file)

    elif style == 'techno':
        print('Chosen style: Techno')
        change_midi_instrument(melody_file, 80)
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

    else:
        print('Chosen style: Custom')
        change_midi_instrument(melody_file, melody_instrument)
        change_midi_instrument(chord_file, chord_instrument)
        merge_melody_with_chords(melody_file, chord_file, song_file)


def add_drum_beats(song_file, beat_style=1):
    # choose beat style
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

# Techno instruments - 80 or 81, 90 or  93
# Strings instruments - 40, 48 / 52, 48


if __name__ == "__main__":
    # for i in range(10):
    # song_post_processing('melody.mid', 'chords.mid', 'song.mid', style='techno')
    generate_song(bar_number=16, style='church', chords=['D:maj', 'A:maj'])

    melody_mid = pretty_midi.PrettyMIDI('../visualizer/app/static/2019-01-09-22-31-17/melody.mid')
    chord_mid = pretty_midi.PrettyMIDI('../visualizer/app/static/2019-01-09-22-31-17/chords.mid')
    melody_pr = melody_mid.get_piano_roll(fs=12)
    chord_pr = chord_mid.get_piano_roll(fs=12)

    pr_length = 384
    melody_pr = melody_pr[:, :pr_length]
    chord_pr = chord_pr[:, :pr_length]
    print(melody_pr.shape)
    melody_midi = piano_roll_to_pretty_midi(melody_pr, fs=12)
    chord_midi = piano_roll_to_pretty_midi(chord_pr, fs=12)
    melody_midi.write('melody.mid')
    chord_midi.write('chords.mid')
    melody_midi = MidiFile('melody.mid')
    chord_midi = MidiFile('chords.mid')
    melody_midi.tracks.append(chord_midi.tracks[-1])
    melody_midi.save('song.mid')

    song_styling('melody.mid', 'chords.mid', 'song.mid', style="techno")




    # os.rename('melody.mid', 'melody-{}.mid'.format(i))
    # os.rename('chords.mid', 'chords-{}.mid'.format(i))
    # os.rename('song.mid', 'song-{}.mid'.format(i))