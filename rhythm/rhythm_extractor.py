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
ABC_FILE ='ashover.abc'


class RhythmExtractor:
    def __init__(self, filename):
        self.train_file = filename

    def extract_rhythm_to_text(self, score):
        notes_list = []
        for i in range(len(score)):
            if type(score[i]) == stream.Part:
                for j in range(len(score[i])):
                    if type(score[i][j]) == stream.Measure:
                        for n in score[i][j].notesAndRests:
                            notes_list.append(n)

        beat_pattern = ''
        for n2 in notes_list:
            num_notes = int(n2.duration.quarterLength / 0.25)   # time step is assumed to be 16th for now
            if type(n2) == note.Note:
                beat_pattern += '-' + '>' * (num_notes - 1)
            else:
                beat_pattern += '_' * num_notes

        return beat_pattern

    def extract_beatlines_for_file(self, is_write_to_file=True):
        beats_lines = []
        abc = converter.parse('ashover.abc')

        # decode abc notation into music21 objects
        # then, encode to our own rhythm notation
        for k in range(len(abc)):
            beat_pattern = self.extract_rhythm_to_text(abc[k])
            beats_lines.append(beat_pattern)

        # padding with rests
        longest_beat = max([len(beats) for beats in beats_lines])
        print('Longest beat in dataset: ' + str(longest_beat))
        for i in range(len(beats_lines)):
            beats_lines[i] = beats_lines[i] + '_' * (longest_beat - len(beats_lines[i]))

        if is_write_to_file:
            beats_file = open('beats_nottingham_training.txt', 'w+')
            _ = [beats_file.write(beats_lines[i] + '\n') for i in range(len(beats_lines))]
            beats_file.close()

        return beats_lines


if __name__ == "__main__":
    rhythm_extractor = RhythmExtractor(ABC_FILE)
    rhythm_extractor.extract_beatlines_for_file()