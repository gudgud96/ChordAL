'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Extracting rhythm information for training using music21.

Improvements needed:
( ) Now, only one opus is used for experiment, extract other opus as well.
( ) Now, only Nottingham dataset is used with abc notation, expand it over midi files
( ) Scrape over abcnotation.com
'''
from music21 import *
from rhythm_generator import show_beats_as_notes


ABC_FILE ='ashover.abc'

def extract_rhythm_to_text(abc_file): 
    beats_lines = []
    abc = converter.parse('ashover.abc')

    # decode abc notation into music21 objects
    # then, encode to our own rhythm notation
    for k in range(len(abc)):
        notes_list = []
        for i in range(len(abc[k])):
            if type(abc[k][i]) == stream.Part:
                for j in range(len(abc[k][i])):
                    if type(abc[k][i][j]) == stream.Measure:
                        for n in abc[k][i][j].notesAndRests:
                            notes_list.append(n)

        beat_pattern = ''
        for n2 in notes_list:
            num_notes = int(n2.duration.quarterLength / 0.25)   # time step is assumed to be 16th
            if type(n2) == note.Note:
                beat_pattern += '-' + '>' * (num_notes - 1)
            else:
                beat_pattern += '_' * num_notes

        beats_lines.append(beat_pattern)

    # padding with rests
    longest_beat = max([len(beats) for beats in beats_lines])
    print('Longest beat in dataset: ' + str(longest_beat))
    beats_file = open('beats_nottingham_training.txt', 'w+')
    for beats in beats_lines:
        beats_file.write(beats + '_' * (longest_beat - len(beats)) + '\n')
    beats_file.close()

if __name__ == "__main__":
    extract_rhythm_to_text(ABC_FILE)