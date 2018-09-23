'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    - Generate chord sequence according to measures.
            - This is to prepare for the note generator, as a music training data needs to have its chords
            and notes extracted, in order to feed them into the note generator.
            - Accuracy / quality of chord extracted directly determines the quality of the note generator.

Improvements needed:
( ) Expand the usage to Lakh dataset and all other pieces in Nottingham dataset
( ) For chord extraction in scraped MIDI, the stream objects are Voices instead of Measures. If so, 
    how long the voice should we take for chord extraction?
( ) Only major chords and settings are considered. Think for minor chords also.
( ) Did not account for modulation and non-tonic scale notes

'''

from music21 import *
import logging
import re
from collections import OrderedDict
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('test.log', mode='w')
handler.setLevel(logging.INFO)
# add the handlers to the logger
logger.addHandler(handler)

measures = []
key_signature = []
key_signature_list = ['gb', 'db', 'ab', 'eb', 'bb', 'f', 'c', 'g', 'd', 'a', 'e', 'b', 'f#']

# to parse a song and understand the structure of its music21 stream object
def parse(a, string, num):
    if num > 1000:
        return
    if hasattr(a, '__len__'):
        logger.info(string + str(a) + ' ' + str(len(a)))
        if type(a) == stream.Measure:
            measures.append(a)
        for i in range(len(a)):
            parse(a[i], string + '--', num + 1)
    else:
        logger.info(string + str(a))
        if type(a) == key.KeySignature:
            key_signature.append(a)


def extract_measures_from_score(score_name):
    # a.show()
    parse(score_name, '', 0)
    notes = [0 for _ in range(len(measures))]
    for measure in measures:
        notes[measure.number] = [elem.name for elem in measure if type(elem) == note.Note]
    return [note for note in notes if note != 0]


def extract_chord_from_notes(measure, pre_chord, key_signature):
    # TODO: did not account for non-tonic scale notes and modulation
    major_scale_chords = get_chord_list_from_key(key_signature[0].sharps)
    possible_chords = []
    # print([note for note in measure])
    if measure:
        for note in measure:
            for i in range(len(major_scale_chords)):
                if note in major_scale_chords[i]:
                    possible_chords.append(i + 1)
        # order by number of counts, then remove duplicates
        # then get the first candidate by brute force for now
        ordered_possible_chords = sorted(possible_chords, key=possible_chords.count, reverse=True)
        ordered_possible_chords = list(OrderedDict.fromkeys(ordered_possible_chords))
        # print(ordered_possible_chords[0:2])
        if ordered_possible_chords:
            return ordered_possible_chords[0]
        else:
            return 0
    else:
        return 0        # in case of empty measures


def get_chord_list_from_key(key):
    key_name = __get_key_name(key)
    tonic_scale = __get_tonic_scale(key_name)
    chord_list = []
    for i in range(len(tonic_scale)):
        chord_list.append('{}{}{}'.format(tonic_scale[i % 7], tonic_scale[(i + 2) % 7], tonic_scale[(i + 4) % 7]))
    return chord_list


def __get_key_name(num_of_sharps):
    num_of_sharps += 6  # offset number of sharps with list indexing
    return key_signature_list[num_of_sharps]


def __get_tonic_scale(key_name):
    sc1 = scale.MajorScale(key_name)
    return [re.findall('[A-Z][b#]?', str(p))[0] for p in sc1.getPitches(key_name + '2', key_name + '3')][:-1]


def __convert_chord_list_to_names(chord_list, key_name):
    tonic_scale = __get_tonic_scale(key_name)
    maj_min_seq = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'min']
    return ' > '.join(['{}:{}'.format(tonic_scale[chord_num - 1], maj_min_seq[chord_num - 1]) for chord_num in chord_list])


def extract_chords(score):      # public method provided to other modules to extract chords
    measures_with_notes = extract_measures_from_score(score)
    chord_list = []
    for measure in measures_with_notes:
        pre_chord = None if not chord_list else chord_list[-1]
        chord = extract_chord_from_notes(measure, pre_chord, key_signature)
        if chord != 0:
            chord_list.append(chord)  # append only if it is not an empty measure
    extracted_chords = __convert_chord_list_to_names(chord_list, __get_key_name(key_signature[0].sharps))
    with open('extracted_chord_sequence.txt', 'a+') as file:
        file.write(extracted_chords + '\n')
    key_signature.clear()


def main():
    open('extracted_chord_sequence.txt', 'w+').write('')        # clear file at each run
    a = converter.parse('ashover.abc')
    # a.show()
    for score in tqdm(a):
        extract_chords(score)


if __name__ == "__main__":
    main()











