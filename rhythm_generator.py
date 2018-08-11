'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    - Train LSTM model on extracted and encoded rhythm data
            - Generate rhythm beats from encoded data by prediction within a 8-quaver timeframe

Improvements needed:
( ) **Fix train-test split as multiple pieces of a certain timeframe
( ) **Fix generate_beats (predict the next beat over a timeframe iteratively)
( ) When the project grows, this script should be written in object-oriented style
( ) Haven't encounter for non-4/4 time signature, semi-quavers, upbeats
( ) LSTM training time-frame is 8-quaver timeframe, note by note generation
'''
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical 
from music21 import *
import numpy as np

# Configurable variables
BEATS_FILE = "beats_nottingham_training.txt"
TT_SPLIT = 4
CHAR_DICT = {
    "-": 1,     # hit note
    ">": 2,     # hold note
    "_": 0      # rest note
}
TIME_STEP = 0.25  # 16th note for our time step
TIME_FRAME = 16   # 16 notes as pre-loaded timeframe for training and prediction
DECODE_DICT = {v: k for k, v in CHAR_DICT.items()}
NUM_CLASSES = len(CHAR_DICT.keys())

# Preprocess data
# Note: do it in numpy array instead of python list to avoid hassles later!
def preprocess_data(filename, tt_split_index = 4):
    beats_lines = [line.rstrip() for line in open(filename, 'r+')]
    beats_processed = np.zeros((len(beats_lines), len(beats_lines[0])))
    for i in range(len(beats_processed)):
        for j in range(len(beats_processed[0])):
            beats_processed[i][j] = CHAR_DICT[beats_lines[i][j]]    # enumeration using dict

    # Generate train and test set
    # Idea: specify a timeframe (set as 16 for now) and predict the next beat after 16 beats
    # Slice each song into pieces of length 17, 16 to training set and 1 to test set
    # Problem with this idea: you may start with '>', which seems a bit absurd, but let's see how
    
    beats_in_tf = []
    for i in range(len(beats_processed)):
        cur_beat = beats_processed[i]
        for j in range(TIME_FRAME, ):

    
    X, Y = beats_processed[:, :-1], beats_processed[:, -1]
    X_oh, Y_oh = to_categorical(X, num_classes=NUM_CLASSES), to_categorical(Y, num_classes=NUM_CLASSES)
    X_train, X_test, Y_train, Y_test = X_oh[:tt_split_index], X_oh[tt_split_index:], \
                                       Y_oh[:tt_split_index],Y_oh[tt_split_index:]
    
    print("X_train.shape: {}, Y_train.shape: {}".format(X_train.shape, Y_train.shape))
    print("X_test.shape:{} , Y_test.shape: {}".format(X_test.shape, Y_test.shape))
    return X_train, X_test, Y_train, Y_test

# Build the model - using layers of LSTM
def build_model(X_train, X_test, Y_train, Y_test):
    num_seq, num_dim = X_train.shape[1], X_train.shape[2]
    
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(num_seq, num_dim)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(num_dim))
    model.add(Activation('softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10)

    scores = model.evaluate(X_train, Y_train, verbose=True)
    print('Train loss:', scores[0])
    print('Train accuracy:', scores[1])
    scores = model.evaluate(X_test, Y_test, verbose=True)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    return model

# Generate beats using prediction
def generate_beats(model):
    predict_beat = '----->->'
    predict_beat_oh = one_hot_encode_beats(predict_beat[:-1])
    prediction = model.predict_classes(predict_beat_oh.reshape(1, predict_beat_oh.shape[0], predict_beat_oh.shape[1]))
    predict_result = predict_beat[:-1] + decode_beats_from_one_hot(prediction)
    print("Predicted: " + predict_result)
    print("Actual: " + predict_beat)
    show_beats_as_notes(predict_result)

def one_hot_encode_beats(beat_pattern):
    encoded_beat_pattern = np.zeros((len(beat_pattern)))
    for i in range(len(beat_pattern)):
        encoded_beat_pattern[i] = CHAR_DICT[beat_pattern[i]]
    return to_categorical(encoded_beat_pattern, num_classes=NUM_CLASSES)

def decode_beats_from_one_hot(one_hot_beats):
    return ''.join([DECODE_DICT[encode] for encode in one_hot_beats.flatten()])

def show_beats_as_notes(beat_pattern):
    cur_quarter_len = 0
    note_stream = stream.Stream()

    for i in range(len(beat_pattern)):
        cur_char = beat_pattern[i]
        
        if cur_char == '-':     # hit note
            if cur_quarter_len > 0:
                note_stream, cur_quarter_len = create_note_to_stream(note_stream, cur_quarter_len, is_note='note')
            cur_quarter_len += TIME_STEP
            if i == len(beat_pattern) - 1:
                note_stream, cur_quarter_len = create_note_to_stream(note_stream, cur_quarter_len, is_note='note')

        elif cur_char == '>':   # hold note
            cur_quarter_len += TIME_STEP
            if i == len(beat_pattern) - 1:
                note_stream, cur_quarter_len = create_note_to_stream(note_stream, cur_quarter_len, is_note='note')

        else:                   # rest note
            if beat_pattern[i-1] != '_':
                note_stream, cur_quarter_len = create_note_to_stream(note_stream, cur_quarter_len, is_note='note')
            cur_quarter_len += TIME_STEP
            if i == len(beat_pattern) - 1 or i < len(beat_pattern) - 1 and beat_pattern[i+1] != '_':
                note_stream, cur_quarter_len = create_note_to_stream(note_stream, cur_quarter_len, is_note='rest')

    note_stream.show()

def create_note_to_stream(stream, quarter_len, is_note='note'):
    if is_note == 'note':
        new_note = note.Note()
    else:
        new_note = note.Rest()
    new_note.duration = duration.Duration(quarter_len)
    stream.append(new_note)
    return stream, 0        # reset cur_quarter_len to 0

# main function
def main():
    X_train, X_test, Y_train, Y_test = preprocess_data(BEATS_FILE, tt_split_index = TT_SPLIT)
    model = build_model(X_train, X_test, Y_train, Y_test)
    generate_beats(model)

if __name__ == "__main__":
    main()