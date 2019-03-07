import sys,os
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))

import os

from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import LSTM, Dense, CuDNNLSTM
import numpy as np
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical

from dataset.data_pipeline import DataPipeline
import matplotlib.pyplot as plt


def get_data():
    dp = DataPipeline()
    chords, melodies = dp.get_csv_nottingham_cleaned()
    
    # get rid of leading and trailing zeros in melodies, and trim chords accordingly
    for i in range(len(melodies)):
        pre_len = len(melodies[i])
        temp_melody = np.trim_zeros(melodies[i], 'f')
        len_change = pre_len - len(temp_melody)
        temp_chords = chords[i][len_change:]         # trim leading zeros

        pre_len = len(temp_melody)
        temp_melody = np.trim_zeros(temp_melody, 'b')
        len_change = pre_len - len(temp_melody)
        temp_chords = temp_chords[:len(temp_chords) - len_change]

        # print(len(temp_chords))

        temp_melody = np.insert(temp_melody, 0, 128)
        temp_melody = np.insert(temp_melody, temp_melody.shape[-1], 129)
        # print(len(temp_melody))

        # padding to ensure the tensors have same length
        if len(temp_melody) < 600:
            melodies[i] = np.pad(temp_melody, (0, 600 - len(temp_melody)), mode='constant', constant_values=0)
        if len(temp_chords) < 600:
            chords[i] = np.pad(temp_chords, (0, 600 - len(temp_chords)), mode='constant', constant_values=0)

    # add start and end of melodies
    #melodies = np.insert(melodies, 0, 128, axis=-1)
    #melodies = np.insert(melodies, melodies.shape[-1], 129, axis=-1)
    #print(melodies.shape)
    melodies_target = np.copy(melodies)[:, 1:]
    melodies_target = np.insert(melodies_target, melodies.shape[-1] - 1, 0, axis=-1)
    return chords, melodies, melodies_target


def main():
    encoder_input_data, decoder_input_data, decoder_target_data = get_data()
    encoder_input_data = to_categorical(encoder_input_data, num_classes=26)
    decoder_input_data = to_categorical(decoder_input_data, num_classes=130)
    decoder_target_data = to_categorical(decoder_target_data, num_classes=130)

    num_encoder_tokens = 26
    num_decoder_tokens = 130    # add start and end token
    latent_dim = 128

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = CuDNNLSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    if not os.path.exists('s2s.h5'):
        optimizer = Adam(lr=0.0001, clipnorm=1)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['categorical_accuracy'])

        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=32,
                  epochs=500,
                  validation_split=0.1)

        plt.plot(range(len(history.history['loss'])), history.history['loss'], label='train loss')
        plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], label='validation loss')

        plt.savefig('loss_train_test.png')

        # Save model
        model.save_weights('s2s.h5')

    else:
        model.load_weights('s2s.h5')

    out = model.predict([np.expand_dims(encoder_input_data[60], axis=0), np.expand_dims(decoder_input_data[60], axis=0)])
    print(np.argmax(out, axis=-1))

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, 128] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = [128]
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, 0, :], axis=-1)
            sampled_char = sampled_token_index
            decoded_sentence.append(sampled_char)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == 129 or
                    len(decoded_sentence) >= 1200):
                stop_condition = True
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
            #print(target_seq)

            # Update states
            states_value = [h, c]

        return decoded_sentence

    input_seq = np.expand_dims(encoder_input_data[60], axis=0)
    res = decode_sequence(input_seq)
    print(res)
    print(len(res))
    print(np.argmax(decoder_input_data[60], axis=-1))
    print(np.argmax(decoder_target_data[60], axis=-1))

if __name__ == "__main__":
    main()
    # get_data()
