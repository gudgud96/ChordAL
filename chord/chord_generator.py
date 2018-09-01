'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Train and generate chord sequence from chord sequence dataset.

Improvements needed:
( ) Data preprocess done. Build model for training and generation.
( ) Do an analysis on the usage of chords at the commented section.
( ) Duplicate code with rhythm generator may need refactoring in preprocess_data.

'''
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical 


# Configurable variables
CHORD_SEQUENCE_FILE = "chord_sequence_file.txt"
TT_SPLIT = 0.8  # percentage of train / test
CHORD_DICT = {
  "Cb": 12, "C": 1, "C#": 2, "Db": 2, "D": 3, "D#": 4, "Eb": 4, "E": 5, "E#": 6,
  "Fb": 5, "F": 6, "F#": 7, "Gb": 7, "G": 8, "G#": 9, "Ab": 9, "A": 10, "A#": 11, 
  "Bb": 11, "B": 12, "B#": 1
}
DECODE_DICT = {
  1: "C", 2: "C#", 3: "D", 4: "D#", 5: "E", 6: "F", 7: "F#", 8: "G", 9: "G#", 10: "A", 11: "A#", 12: "B"
}
MAJ, MIN = ["maj", "min"]   # major is even and minor is +1 odd. Hence Cmaj=2, Cmin=3, C#maj=4 and etc.
TIME_FRAME = 9              # 8 notes as pre-loaded timeframe for training, 1 for prediction
NUM_CLASSES = 26            # 2-25 for major and minor, 0 for None, 1 is not used

# Preprocess data
def preprocess_data(filename, tt_split = 0.8):
    chord_sequences = [a.rstrip() for a in open(filename, 'r+').readlines()]
    len_list = [len(s.split(' > ')) for s in chord_sequences]
    # plt.hist(len_list, normed=True, bins=30)    # from this histogram, median is 210 and max is 400+. take 300 as max.
    # plt.show()
    max_len = 300
    chords_processed = np.zeros((len(chord_sequences), max_len))
    
    for i in range(len(chord_sequences)):
        sequence = chord_sequences[i]
        chords = sequence.split(' > ')
        chords = chords[:] if len(chords) <= max_len else chords[:max_len]
        for j in range(len(chords)):
            chords_processed[i][j] = chord_to_id(chords[j])
    
    # print(chord_sequences[12])
    # print(chords_processed[12])

    # TODO: do an analysis of usage of chords
    
    # same strategy as rhythm generator, but tf=8
    # needs to reconsider if this is a good strategy, because global structure will be lost.
    chords_in_tf = []
    for i in range(len(chords_processed)):
        cur_chords, cur_chords_len = chords_processed[i], len(chords_processed[i])
        for j in range(cur_chords_len - TIME_FRAME + 1):
            chords_in_tf.append([cur_chords[j : j + TIME_FRAME]])
        
    chords_in_tf = np.squeeze(np.asarray(chords_in_tf))
    # print(chords_in_tf[:10])
    print("chords_in_tf.shape : {}".format(chords_in_tf.shape))
    X, Y = chords_in_tf[:, :-1], chords_in_tf[:, -1]  
    X_oh, Y_oh = to_categorical(X, num_classes=NUM_CLASSES), to_categorical(Y, num_classes=NUM_CLASSES)

    tt_split_index = int(tt_split * len(chords_in_tf))
    X_train, X_test, Y_train, Y_test = X_oh[:tt_split_index], X_oh[tt_split_index:], \
                                       Y_oh[:tt_split_index],Y_oh[tt_split_index:]
    
    print("X_train.shape: {}, Y_train.shape: {}".format(X_train.shape, Y_train.shape))
    print("X_test.shape:{} , Y_test.shape: {}".format(X_test.shape, Y_test.shape))
    return X_train, X_test, Y_train, Y_test

# change chord of format <name>:<tonality> to id between 2-25
def chord_to_id(chord):
    if ':' not in chord:
        return 0
    chord_name, chord_tonality = chord.split(':')
    chord_id = CHORD_DICT[chord_name] * 2           # leave 0 for empty chords
    if MIN in chord_tonality:
        chord_id += 1                               # odd number for minors
    return chord_id

def id_to_chord(idnum):
    if idnum == 0:
        return "-"
    elif idnum % 2 == 0:
        return DECODE_DICT[idnum / 2] + MAJ
    else:
        return DECODE_DICT[(idnum - 1) / 2] + MIN

def chord_to_id_test():
    assert chord_to_id("A#:maj") == 22
    assert chord_to_id("C:min") == 3
    assert chord_to_id("Gb:maj(9)") == 14
    assert chord_to_id("D:sus") == 6
    assert chord_to_id("F:min(5)") == 13

def id_to_chord_test():
    assert id_to_chord(22) == "A#maj"
    assert id_to_chord(3) == "Cmin"
    assert id_to_chord(14) == "F#maj"
    assert id_to_chord(6) == "Dmaj"
    assert id_to_chord(13) == "Fmin" 





def main():
    preprocess_data(CHORD_SEQUENCE_FILE, TT_SPLIT)
    # chord_to_id_test()
    # id_to_chord_test()
    

main()