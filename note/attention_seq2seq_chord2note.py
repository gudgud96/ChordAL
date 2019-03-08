import sys,os


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


def main():
    encoder_input_data, decoder_input_data, decoder_target_data = get_data()
    encoder_input_data = to_categorical(encoder_input_data, num_classes=26)
    decoder_input_data = to_categorical(decoder_input_data, num_classes=130)
    decoder_target_data = to_categorical(decoder_target_data, num_classes=130)

    print(encoder_input_data.shape, decoder_input_data.shape, decoder_target_data.shape)

    num_encoder_tokens = 26
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

    # Run training
    if not os.path.exists('s2s_attention.h5'):
        optimizer = Adam(clipnorm=1.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['categorical_accuracy'])

        print(model.summary())

        history = model.fit(x=[encoder_input_data, decoder_input_data], y=[decoder_target_data],
                  batch_size=32,
                  epochs=1,
                  validation_split=0.1)

        plt.plot(range(len(history.history['loss'])), history.history['loss'], label='train loss')
        plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], label='validation loss')

        plt.savefig('loss_train_test.png')

        # Save model
        model.save_weights('s2s_attention.h5')

    else:
        model.load_weights('s2s_attention.h5')

    # def decode_sequence(input_seq):
    #     decoder_input = np.zeros(shape=(max_length, num_decoder_tokens))
    #     decoder_input[0, 128] = 1.
    #     for i in range(1, 100):
    #         print(np.argmax(decoder_input, axis=-1)[:10])
    #         output = model.predict([input_seq, np.expand_dims(decoder_input, axis=0)]).argmax(axis=-1)
    #         output_char = output[0, i]
    #         print("output_char: ", output_char)
    #         decoder_input[i, output_char] = 1.
    #     return np.argmax(decoder_input, axis=-1)
    #
    ind = 10
    input_seq = np.expand_dims(encoder_input_data[ind], axis=0)
    visualize_attention(model, input_seq)
    # res = decode_sequence(input_seq)
    # print(res)


def visualize_attention(model, input_seq):
    attention_layer = model.get_layer('attention')  # or model.layers[7]
    attention_model = Model(inputs=model.inputs, outputs=model.outputs + [attention_layer.output])

    print(attention_model)
    print(attention_model.output_shape)

    def attent_and_generate(input_seq, num_decoder_tokens):
        decoder_input = np.zeros(shape=(40, num_decoder_tokens))
        decoder_input[0, 128] = 1.

        for i in range(1, 40):
            output, attention = attention_model.predict([input_seq, np.expand_dims(decoder_input, axis=0)])
            output_char = np.argmax(output[0, i], axis=-1)
            print(output_char)
            decoder_input[i, output_char] = 1.
            attention_density = attention[0]

        return attention_density, np.argmax(decoder_input, axis=-1)

    def visualize(input_seq):
        input_seq = input_seq[:, 6:]
        attention_density, notes = attent_and_generate(input_seq, num_decoder_tokens=130)
        print(attention_density.shape)
        plt.clf()
        print(np.argmax(input_seq, axis=-1))
        print(attention_density[:len(notes), :len(input_seq[0]) + 1].shape)

        cmap = seaborn.cm.rocket_r
        ax = seaborn.heatmap(attention_density[:len(notes), :len(notes)],
                             xticklabels=[w for w in np.argmax(input_seq, axis=-1)[0][:len(notes)]],
                             yticklabels=[w for w in notes],
                             cmap=cmap)

        ax.invert_yaxis()
        plt.show()

    visualize(input_seq)


if __name__ == "__main__":
    main()
    # get_data()