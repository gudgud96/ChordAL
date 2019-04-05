import os
from collections import Counter

from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import LSTM, Dense
import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical
from tqdm import tqdm

from dataset.data_pipeline import DataPipeline
import matplotlib.pyplot as plt


def get_data():
    dp = DataPipeline()
    chords, melodies = dp.get_csv_nottingham_cleaned()

    counters = [len(Counter(melodies[k]).keys()) for k in range(len(melodies))]
    ind = counters.index(1)
    print(melodies[ind])

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

        temp_melody = np.insert(temp_melody, 0, 128)
        temp_melody = np.insert(temp_melody, temp_melody.shape[-1], 129)

        # padding to ensure the tensors have same length
        if len(temp_melody) < 600:
            melodies[i] = np.pad(temp_melody, (0, 600 - len(temp_melody)), mode='constant', constant_values=0)
        if len(temp_chords) < 600:
            chords[i] = np.pad(temp_chords, (0, 600 - len(temp_chords)), mode='constant', constant_values=0)

    melodies_target = np.copy(melodies)
    melodies_target = melodies_target[:, 1:]
    melodies_target = np.insert(melodies_target, melodies_target.shape[-1], 129, axis=-1)
    print(chords.shape, melodies.shape, melodies_target.shape)
    return chords, melodies, melodies_target


def beam_search(input_seq, encoder_model, decoder_model, num_decoder_tokens, beam_width=10):
    print("Beam search")
    candidate_list = [[128]]
    pre_dict = {(128,): 1}

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    for i in tqdm(range(600)):
        new_candidate_list = []
        cur_dict = {}

        for k in candidate_list:
            input_array = np.zeros((1, 601, num_decoder_tokens))
            for index, x in enumerate(k):
                input_array[0, index, x] = 1.

            output_tokens, h, c = decoder_model.predict([input_array] + states_value)
            beam_indices = np.argpartition(output_tokens[0, len(k), :], -beam_width)[-beam_width:]
            # print(beam_indices)
            for ind in beam_indices:
                temp = np.append(k, ind)
                # print(temp)
                new_candidate_list.append(temp)
                cur_dict[tuple(temp)] = output_tokens[0, 0, ind] * pre_dict[tuple(k)]

        # print(cur_dict)
        candidate_list = sorted(cur_dict, key=cur_dict.get, reverse=True)[:beam_width]
        candidate_list = [list(k) for k in candidate_list]
        # print("Candidates this round: {}".format(candidate_list))
        pre_dict = cur_dict

    return candidate_list, pre_dict


def decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    decoder_input = np.zeros(shape=(600, num_decoder_tokens))
    decoder_input[0, 128] = 1.
    for i in tqdm(range(1, 600)):
        # print("input: ", np.argmax(decoder_input, axis=-1)[:10])
        temp = np.expand_dims(decoder_input, axis=0)
        output, _, _ = decoder_model.predict([temp] + states_value)
        output = np.argmax(output, axis=-1)
        # print("output: ", output[0][:10])
        output_char = output[0, i]
        # print("output_char: ", output_char)
        decoder_input[i, output_char] = 1.
    return np.argmax(decoder_input, axis=-1)


def main():
    encoder_input_data, decoder_input_data, decoder_target_data = get_data()
    encoder_input_data = to_categorical(encoder_input_data, num_classes=26)
    decoder_input_data = to_categorical(decoder_input_data, num_classes=130)
    decoder_target_data = to_categorical(decoder_target_data, num_classes=130)

    num_encoder_tokens = 26
    num_decoder_tokens = 130
    num_duration = 50
    latent_dim = 128

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    if not os.path.exists('s2s_note.h5'):
        optimizer = Adam(clipnorm=1.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics='categorical_accuracy')

        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=32,
                  epochs=5,
                  validation_split=0.1)

        plt.plot(range(len(history.history['loss'])), history.history['loss'], label='train loss')
        plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], label='validation loss')

        plt.savefig('loss_train_test.png')

        # Save model
        model.save_weights('s2s_note.h5')

    else:
        model.load_weights('s2s_fariz.h5')

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

    def decode(input_seq, beam_width=3):
        if beam_width > 1:
            candidate_list, _ = beam_search(input_seq, encoder_model, decoder_model,
                                                     num_decoder_tokens, beam_width=beam_width)
        else:
            candidate_list = decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens)

        return candidate_list

    ind = 10
    input_seq = np.expand_dims(encoder_input_data[ind], axis=0)
    beam_width = 1
    res = decode(input_seq, beam_width=beam_width)
    if beam_width == 1:
        print(res)
    else:
        for r in res:
            print("res: ", r)
            print(len(r))
    print(np.argmax(decoder_input_data, axis=-1)[ind][:40])
    print(np.argmax(decoder_target_data, axis=-1)[ind][:40])

    for i in range(1, 15, 2):
        res2 = model.predict([np.expand_dims(encoder_input_data[ind], axis=0),
                             np.expand_dims(decoder_input_data[ind][:i], axis=0)])
        print("Input: {}".format(np.argmax(decoder_input_data[ind][:i], axis=-1)))
        print("Output: {}".format(np.argmax(res2, axis=-1)))
        print()


if __name__ == "__main__":
    main()
    # get_data()