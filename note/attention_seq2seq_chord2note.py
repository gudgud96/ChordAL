import sys,os
import time

from tqdm import tqdm

sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import os

from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import LSTM, Dense, dot, Activation, concatenate, TimeDistributed
import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical

from dataset.data_pipeline import DataPipeline
import matplotlib.pyplot as plt
import seaborn
import pretty_midi
from utils import piano_roll_to_pretty_midi, merge_melody_with_chords, chord_index_to_piano_roll


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

        #print(len(temp_chords))

        temp_melody = np.insert(temp_melody, 0, 128)
        temp_melody = np.insert(temp_melody, temp_melody.shape[-1], 129)
        #print(len(temp_melody))

        # padding to ensure the tensors have same length
        if len(temp_melody) < 600:
            melodies[i] = np.pad(temp_melody, (0, 600 - len(temp_melody)), mode='constant', constant_values=0)
        if len(temp_chords) < 600:
            chords[i] = np.pad(temp_chords, (0, 600 - len(temp_chords)), mode='constant', constant_values=0)

    melodies_target = np.copy(melodies)
    melodies_target = melodies_target[:, 1:]
    melodies_target = np.insert(melodies_target, melodies_target.shape[-1], 0, axis=-1)
    print(chords.shape, melodies.shape, melodies_target.shape)
    return chords, melodies, melodies_target


def beam_search(input_seq, encoder_model, decoder_model, num_decoder_tokens, beam_width=10):
    print("Beam search...")
    candidate_list = [[128]]
    pre_dict = {(128,): 1}
    length = 40

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    for _ in tqdm(range(length)):
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


def main():
    encoder_input_data, decoder_input_data, decoder_target_data = get_data()
    from utils import convert_chord_indices_to_embeddings

    # prepare embeddings
    encoder_input_embeddings = np.array([convert_chord_indices_to_embeddings(chord) for chord in encoder_input_data])
    # encoder_input_data = to_categorical(encoder_input_data, num_classes=26)
    decoder_input_data = to_categorical(decoder_input_data, num_classes=130)
    decoder_target_data = to_categorical(decoder_target_data, num_classes=130)

    print(encoder_input_data.shape, decoder_input_data.shape, decoder_target_data.shape)

    num_encoder_tokens = 32
    num_decoder_tokens = 130
    latent_dim = 128
    max_length = 600

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    print("Encoder inputs: ", encoder_inputs)
    encoder = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    print("Encoder outputs: ", encoder_outputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    print("Decoder inputs: ", decoder_inputs)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=[state_h, state_c])
    print("Decoder outputs: ", decoder_outputs)

    # Added attention layer here
    attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)
    print('attention', attention)

    context = dot([attention, encoder_outputs], axes=[2, 1])
    print('context', context)

    decoder_combined_context = concatenate([context, decoder_outputs])
    print('decoder_combined_context', decoder_combined_context)

    # Has another weight + tanh layer as described in equation (5) of the paper
    output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)
    output = TimeDistributed(Dense(130, activation="softmax"))(output)
    print('output', output)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[output])
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)
    print('attention', attention)

    context = dot([attention, encoder_outputs], axes=[2, 1])
    print('context', context)

    decoder_combined_context = concatenate([context, decoder_outputs])
    print('decoder_combined_context', decoder_combined_context)

    # Has another weight + tanh layer as described in equation (5) of the paper
    output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)
    output = TimeDistributed(Dense(130, activation="softmax"))(output)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # Run training using Xavier's learning rate manual decay mechanism
    if not os.path.exists('s2s_attention_augmented.h5'):

        def train_by_adaptive_lr():
            # manual run epochs, change decay rate for optimizers each epoch
            number_of_epochs = 10
            prev_validation_loss = 10000
            cur_validation_loss = 10000
            losses = []
            val_losses = []

            # run for 1 epoch first
            learning_rate = 1.0001
            decay_factor = 0.9
            optimizer = Adam(clipnorm=1.0, lr=learning_rate)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                          metrics=['categorical_accuracy'])
            print(model.summary())  # only print on first epoch

            history = model.fit(x=[encoder_input_data[:100], decoder_input_data[:100]], y=[decoder_target_data[:100]],
                                batch_size=32,
                                epochs=1,
                                validation_split=0.1)

            cur_validation_loss = history.history['val_loss'][0]

            print("Loss: {} Validation loss: {}\n".format(history.history['loss'][0], history.history['val_loss'][0]))
            losses.append(history.history['loss'][0])
            val_losses.append(history.history['val_loss'][0])

            for ep in range(number_of_epochs):
                optimizer = Adam(clipnorm=1.0, lr=learning_rate)
                model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                              metrics=['categorical_accuracy'])

                history = model.fit(x=[encoder_input_data[:100], decoder_input_data[:100]], y=[decoder_target_data[:100]],
                          batch_size=32,
                          epochs=1,
                          validation_split=0.1)

                prev_validation_loss = cur_validation_loss
                cur_validation_loss = history.history['val_loss'][0]

                validation_delta_percentage = (cur_validation_loss - prev_validation_loss) / prev_validation_loss
                print("validation delta percentage: {}".format(validation_delta_percentage))
                print("Loss: {} Validation loss: {}\n".format(history.history['loss'][0], history.history['val_loss'][0]))

                if abs(validation_delta_percentage) < 0.01:
                    learning_rate *= decay_factor

                losses.append(history.history['loss'][0])
                val_losses.append(history.history['val_loss'][0])
            return losses, val_losses

        optimizer = Adam(clipnorm=1.0, lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['categorical_accuracy'])
        print(model.summary())  # only print on first epoch

        def train_each_timestep(encoder_input_data, decoder_input_data, decoder_target_data):
            epochs = 5
            length_of_sequence = 100
            train_test_split = 0.1
            losses = []
            val_losses = []

            # smaller dataset
            encoder_input_data = encoder_input_data[:100]
            decoder_input_data = decoder_input_data[:100]
            decoder_target_data = decoder_target_data[:100]

            num_examples = len(encoder_input_data)
            split_point = int(num_examples * (1-train_test_split))

            encoder_input_train, encoder_input_test = encoder_input_data[:split_point], \
                                                      encoder_input_data[split_point:]
            decoder_input_train, decoder_input_test = decoder_input_data[:split_point], \
                                                      decoder_input_data[split_point:]
            decoder_target_train, decoder_target_test = decoder_target_data[:split_point], \
                                                        decoder_target_data[split_point:]

            for ep in range(epochs):
                t1 = time.time()
                temp_loss = []
                temp_val_loss = []
                for length in range(1, length_of_sequence):
                    print("Sequence index: {}".format(length))
                    history = model.fit(x=[encoder_input_train[:, :length, :], decoder_input_train[:, :length, :]],
                                        y=[decoder_target_train[:, :length, :]],
                                        batch_size=32,
                                        epochs=1)
                    temp_loss.append(history.history["loss"][0])
                    val_loss = model.evaluate(x=[encoder_input_test[:, :length, :], decoder_input_test[:, :length, :]],
                                              y=[decoder_target_test[:, :length, :]])[0]
                    temp_val_loss.append(val_loss)

                losses.append(sum(temp_loss) / len(temp_loss))
                val_losses.append(sum(temp_val_loss) / len(temp_val_loss))
                print("Loss: {} Val loss: {}".format(sum(temp_loss) / len(temp_loss), sum(temp_val_loss) / len(temp_val_loss)))
                print("Time used for 1 epoch: {}".format(time.time() - t1))

            return losses, val_losses

        def normal_training():
            nb_epochs = 250
            history = model.fit(x=[encoder_input_data, decoder_input_data], y=[decoder_target_data],
                                batch_size=32,
                                epochs=nb_epochs,
                                validation_split=0.1)

            losses = history.history['val_loss'][0]
            val_losses = history.history['val_loss'][0]
            return losses, val_losses

        losses, val_losses = normal_training()      # choose training method here

        plt.plot(range(len(losses)), losses, label='train loss')
        plt.plot(range(len(val_losses)), val_losses, label='validation loss')

        plt.savefig('loss_train_test.png')

        # Save model
        model.save_weights('s2s_attention.h5')

    else:
        model.load_weights('s2s_attention_augmented.h5')

    ind = 12
    input_seq = np.expand_dims(encoder_input_embeddings[ind], axis=0)
    print(np.argmax(decoder_target_data[ind], axis=-1))
    visualize_attention(model, input_seq, encoder_input_data[ind], decoder_input_data[ind])
    # res = decode_sequence(input_seq)
    # print(res)


def visualize_attention(model, input_seq, encoder_input_data, decoder_input_data):
    attention_layer = model.get_layer('attention')  # or model.layers[7]
    attention_model = Model(inputs=model.inputs, outputs=model.outputs + [attention_layer.output])

    print(attention_model)
    print(attention_model.output_shape)

    def attent_and_generate(input_seq, decoder_data_seq, num_decoder_tokens):
        length = 100
        decoder_input = np.zeros(shape=(length, num_decoder_tokens))
        decoder_input[0, 128] = 1.
        output_song = []

        for i in range(1, length):
            output, attention = attention_model.predict([input_seq, np.expand_dims(decoder_input, axis=0)])
            # output, attention = attention_model.predict([input_seq, np.expand_dims(decoder_data_seq, axis=0)])
            print("output", np.argmax(output[0], axis=-1))
            output_char = np.argmax(output[0, i], axis=-1)
            print(output_char)
            output_song.append(output_char)
            decoder_input[i, output_char] = 1.
            attention_density = attention[0]

        print(output_song)
        print(attention_density)
        return attention_density, output_song

    def visualize(input_seq, encoder_data_seq, decoder_data_seq):
        print(input_seq.shape)
        print("encoder: ", encoder_data_seq.shape)
        attention_density, notes = attent_and_generate(input_seq, decoder_data_seq, num_decoder_tokens=130)
        print(attention_density.shape)
        plt.clf()
        print(attention_density[:len(notes), :len(input_seq[0]) + 1].shape)

        cmap = seaborn.cm.rocket_r
        ax = seaborn.heatmap(attention_density[:len(notes), :len(notes)],
                             xticklabels=[w for w in encoder_data_seq[:len(notes)]],
                             yticklabels=[w for w in notes],
                             cmap=cmap)

        ax.invert_yaxis()
        plt.show()
        notes = np.transpose(to_categorical(notes, num_classes=128), (1,0))
        print(notes.shape)
        plt.imshow(notes)
        plt.show()
        notes[notes == 1] = 90
        pm = piano_roll_to_pretty_midi(notes, fs=12)
        cm = piano_roll_to_pretty_midi(chord_index_to_piano_roll(encoder_data_seq[:len(notes[0])]), fs=12)
        pm.write("melody_attention.mid")
        cm.write("chord_attention.mid")
        merge_melody_with_chords("melody_attention.mid", "chord_attention.mid", "song_attention.mid")

    visualize(input_seq, encoder_input_data, decoder_input_data)


if __name__ == "__main__":
    main()
    # get_data()