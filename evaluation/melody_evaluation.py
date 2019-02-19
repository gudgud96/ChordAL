import os
from pretty_midi import pretty_midi
from note.chord_to_note_generator import ChordToNoteGenerator
from chord.chord_generator import ChordGenerator
from dataset.data_pipeline import DataPipeline
from music21 import scale
import numpy as np

from utils import piano_roll_to_pretty_midi

cg = ChordGenerator()
scale_cache = {}


def form_chord_array(chords):
    chord_lst = chords.split(' > ')
    return np.array([cg.chord_to_id(chord) for chord in chord_lst])


def get_scale_notes(chord_index):
    if chord_index in scale_cache:
        return scale_cache[chord_index]
    else:
        chord, tonality = cg.id_to_chord(chord_index).split(':')
        print("Key: {} Index: {}".format(chord + ':' + tonality, chord_index))
        if tonality == 'maj':
            sc = scale.MajorScale(chord)
        else:
            sc = scale.MinorScale(chord)
        res = [p.midi for p in sc.getPitches(chord + '2', chord + '5')]     # across 3 octaves
        scale_cache[chord_index] = res
        return res


def evaluate_repeat(notes):
    length_lst = []
    prev = notes[0]
    cur_length = 1
    for i in range(1, len(notes)):
        if notes[i] == prev:
            cur_length += 1
        else:
            length_lst.append(cur_length)
            cur_length = 1
        prev = notes[i]
    length_lst.append(cur_length)
    return length_lst


def evaluate_outliers(notes, chord_array):
    outliers_count = 0
    for i in range(len(chord_array)):
        chord_index = chord_array[i]
        sc = get_scale_notes(chord_index)
        if notes[i] not in sc:
            print("outliers: chord {} -- note {}".format(chord_index, notes[i]))
            outliers_count += 1
    return outliers_count / len(chord_array)


def evaluate_notes(notes, chord_array):
    res = {}
    repeat_length = evaluate_repeat(notes)
    average_repeat = sum(repeat_length) / len(repeat_length)
    longest_repeat = max(repeat_length)
    outlier_score = evaluate_outliers(notes, chord_array)

    res['average_repeat'] = average_repeat
    res['longest_repeat'] = longest_repeat
    res['outlier_ratio'] = outlier_score
    return res


def get_chord_samples(chord_array):
    # add 12 times to fit FS
    res = []
    for chord in chord_array:
        for _ in range(12):
            res.append(chord)
    return res


def convert_midi_to_chord_indices(chord_sample_name):
    dp = DataPipeline()
    temp_chord_midi = pretty_midi.PrettyMIDI(chord_sample_name)
    pr = temp_chord_midi.get_piano_roll(fs=12)
    chord_indices = dp.convert_chord_to_indices(pr)

    # remove leading zeros
    while chord_indices[0] == 0:
        chord_indices = chord_indices[1:]

    return chord_indices


def move_to_note_evaluation_folder():
    folder_name = int(os.listdir('./note_evaluation_results')[-1]) + 1
    os.mkdir('./note_evaluation_results/' + str(folder_name) + '/')
    for i in range(1, 7):
        os.rename('melody_{}.mid'.format(i), 'note_evaluation_results/' + str(folder_name)
                  + '/melody_{}.mid'.format(i))
    os.rename('note_evaluation.txt', 'note_evaluation_results/' + str(folder_name)
              + '/note_evaluation.txt')
    os.remove('melody.mid')


def main():
    evaluation_text = 'note_evaluation.txt'
    open(evaluation_text, 'w+').write('')
    e = open(evaluation_text, 'a+')
    total_res = {}

    model_name = 'attention'            # change this line to swap different model name

    ctng = ChordToNoteGenerator()
    ctng.load_model(model_name, is_fast_load=True)

    f = os.listdir('chord_samples/')
    counter = 1

    for chords in f:
        print('\nEvaluating file: {}...'.format(chords))

        if model_name == 'bidem' or model_name == 'attention':
            chord_array = convert_midi_to_chord_indices('chord_samples/' + chords)
            chord_array_length = len(chord_array)
            tmp = np.pad(chord_array, (0, 1200 - chord_array_length), mode='constant', constant_values=0)
        else:
            chord_array = pretty_midi.PrettyMIDI('chord_samples/' + chords)
            pr = chord_array.get_piano_roll(fs=12)
            chord_array_length = pr.shape[-1]
            tmp = np.pad(pr, [(0, 0), (0, 1200 - chord_array_length)], mode='constant', constant_values=0)

        notes = ctng.generate_notes_from_chord(tmp, is_return=True, is_bidem=(model_name == 'bidem'
                                                                              or model_name == 'attention'))

        melody_pr = pretty_midi.PrettyMIDI('melody.mid').get_piano_roll(fs=12)[:, :chord_array_length]
        melody_midi = piano_roll_to_pretty_midi(melody_pr, fs=12)
        melody_midi.write('melody_{}.mid'.format(counter))
        counter += 1

        notes = np.argmax(notes, axis=0)[:chord_array_length]

        if model_name != 'bidem' and model_name != 'attention':
            chord_array = convert_midi_to_chord_indices('chord_samples/' + chords)      # for evaluation

        res = evaluate_notes(notes, chord_array)
        e.write(chords + "\n")
        for k in res.keys():
            e.write("{}: {}\n".format(k, res[k]))
            if k in total_res:
                total_res[k] += res[k]
            else:
                total_res[k] = res[k]
        e.write("\n")

    e.write('\n-- Average --\n')
    for k in total_res.keys():
        e.write("{}: {}\n".format(k, total_res[k] / len(f)))
    e.close()

    move_to_note_evaluation_folder()


if __name__ == "__main__":
    main()