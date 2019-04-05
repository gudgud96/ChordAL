import pickle
import numpy as np
import pretty_midi
from mido import MidiFile


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def chord_index_to_piano_roll(chord_indices):
    index_to_pitch_map = {}
    for i in range(1, 10):
        index_to_pitch_map[i] = 47 + i
    index_to_pitch_map[10] = 45
    index_to_pitch_map[11] = 46
    index_to_pitch_map[12] = 47

    res = np.full((1200,128), 0)

    for i in range(len(chord_indices)):
        ind = chord_indices[i]
        if ind == 0:
            continue
        base_note = index_to_pitch_map[ind // 2]
        if ind % 2 == 0:    # major
            res[i][base_note] = 90
            res[i][base_note + 4] = 90
            res[i][base_note + 7] = 90
        else:               # minor
            res[i][base_note] = 90
            res[i][base_note + 3] = 90
            res[i][base_note + 7] = 90

    return np.transpose(res, (1,0))


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


def convert_chord_indices_to_embeddings(chords):
    res = []
    pickle_in = open("../dataset/chord_embeddings_dict.pickle", 'rb')
    embeddings_dict = pickle.load(pickle_in)
    embeddings_dict[0] = np.zeros((32,))

    for chord in chords:
        res.append(embeddings_dict[chord])

    return np.array(res)

