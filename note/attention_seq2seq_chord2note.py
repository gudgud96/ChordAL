import sys,os
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
import os
import time
from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import CuDNNLSTM, Dense, dot, Activation, concatenate, TimeDistributed
import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical

from dataset.data_pipeline import DataPipeline
import matplotlib.pyplot as plt


def get_data(preload_embeddings=True):
    dp = DataPipeline()
    chords, melodies = dp.get_csv_nottingham_cleaned()
    res_chords, res_melodies = [], []

    # get rid of leading and trailing zeros in melodies, and trim chords accordingly
    for i in range(len(melodies)):
        pre_len = len(melodies[i])
        temp_melody = np.trim_zeros(melodies[i], 'f')
        len_change = pre_len - len(temp_melody)
        temp_chords = chords[i][len_change:]         # trim leading zeros
        # print(temp_chords)
        # print(temp_melody)
        
        pre_len = len(temp_melody)
        temp_melody = np.trim_zeros(temp_melody, 'b')
        len_change = pre_len - len(temp_melody)
        temp_chords = temp_chords[:len(temp_chords) - len_change]

        # print(len(temp_chords))
        temp_melody = np.insert(temp_melody, 0, 128)
        temp_melody = np.insert(temp_melody, temp_melody.shape[-1], 129)

        # padding to ensure the tensors have same length
        if len(temp_melody) < 602:
            temp_melody = np.pad(temp_melody, (0, 602 - len(temp_melody)), mode='constant', constant_values=0)
        if len(temp_chords) < 602:
            temp_chords = np.pad(temp_chords, (0, 602 - len(temp_chords)), mode='constant', constant_values=0)

        res_chords.append(temp_chords)
        res_melodies.append(temp_melody)

    # new chords and melodies with (1) zeros trimming and (2) add start and end char
    res_chords, res_melodies = np.array(res_chords), np.array(res_melodies)

    melodies_target = np.copy(res_melodies)
    melodies_target = melodies_target[:, 1:]
    melodies_target = np.insert(melodies_target, melodies_target.shape[-1], 0, axis=-1)

    print(res_chords.shape, res_chords.shape, melodies_target.shape)
    return res_chords, res_chords, melodies_target


def main():
    preload_embeddings = False
    encoder_input_data, decoder_input_data, decoder_target_data = get_data()

    if not preload_embeddings:
        encoder_input_data = to_categorical(encoder_input_data, num_classes=26)
        pass
    else:
        from utils import convert_chord_indices_to_embeddings
        encoder_input_data = np.array([convert_chord_indices_to_embeddings(chord) for chord in encoder_input_data])

    # decoder_input_data = to_categorical(decoder_input_data, num_classes=130)
    # decoder_target_data = to_categorical(decoder_target_data, num_classes=130)
    
    # for sparse categorical crossentropy
    # encoder_input_data = np.expand_dims(encoder_input_data, axis=-1)
    # decoder_input_data = np.expand_dims(decoder_input_data, axis=-1)
    # decoder_target_data = np.expand_dims(decoder_target_data, axis=-1)

    print("After encoding: ", encoder_input_data.shape, decoder_input_data.shape, decoder_target_data.shape)
    
    num_encoder_tokens = 26 if not preload_embeddings else 32
    num_decoder_tokens = 130
    latent_dim = 128
    max_length = None

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(max_length, num_encoder_tokens))
    print("Encoder inputs: ", encoder_inputs)
    encoder = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    print("Encoder outputs: ", encoder_outputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(max_length, num_decoder_tokens))
    print("Decoder inputs: ", decoder_inputs)

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
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

        # optimizers and model summary
        optimizer = Adam(clipnorm=1.0, lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['categorical_accuracy'])
        print(model.summary())  # only print on first epoch

        # different types of training methods
        def train_by_adaptive_lr():
            # manual run epochs, change decay rate for optimizers each epoch
            number_of_epochs = 100
            prev_validation_loss = 10000
            cur_validation_loss = 10000
            losses = []
            val_losses = []

            # run for 1 epoch first
            learning_rate = 0.01
            decay_factor = 0.9
            optimizer = Adam(clipnorm=1.0, lr=learning_rate)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                          metrics=['sparse_categorical_accuracy'])
            print(model.summary())  # only print on first epoch

            history = model.fit(x=[encoder_input_data, decoder_input_data], y=[decoder_target_data],
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

                history = model.fit(x=[encoder_input_data, decoder_input_data], y=[decoder_target_data],
                          batch_size=32,
                          epochs=1,
                          validation_split=0.1)

                prev_validation_loss = cur_validation_loss
                cur_validation_loss = history.history['val_loss'][0]

                validation_delta_percentage = (cur_validation_loss - prev_validation_loss) / prev_validation_loss
                print("validation delta percentage: {}".format(validation_delta_percentage))
                print("Loss: {} Validation loss: {}\n".format(history.history['loss'][0], history.history['val_loss'][0]))

                if abs(validation_delta_percentage) < 0.01:
                    print("Decaying...")
                    learning_rate *= decay_factor

                losses.append(history.history['loss'][0])
                val_losses.append(history.history['val_loss'][0])
            return losses, val_losses

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
            nb_epochs = 100
            history = model.fit(x=[encoder_input_data, decoder_input_data], y=[decoder_target_data],
                                batch_size=32,
                                epochs=nb_epochs,
                                validation_split=0.1)

            losses = history.history['loss']
            val_losses = history.history['val_loss']
            return losses, val_losses

        def train_with_generator():
            def generate_preprocessed_data():
                while 1:
                    for i in range(len(encoder_input_data)):
                        input_chord, decoder_input, decoder_target = encoder_input_data[i], \
                                                                     decoder_input_data[i], \
                                                                     decoder_target_data[i]
                        decoder_input = to_categorical(decoder_input, num_classes=130)
                        decoder_target = to_categorical(decoder_target, num_classes=130)
                        yield ([np.expand_dims(input_chord, axis=0), np.expand_dims(decoder_input, axis=0)], np.expand_dims(decoder_target, axis=0))

            history = model.fit_generator(generate_preprocessed_data(),
                                          steps_per_epoch=1458, epochs=3) # this means using all samples 46656, and batch size = 32
            losses = history.history['loss']
            val_losses = history.history['val_loss']
            return losses, val_losses

        losses, val_losses = train_with_generator()      # choose training method here
        # losses, val_losses = train_by_adaptive_lr()
        plt.plot(range(len(losses)), losses, label='train loss')
        plt.plot(range(len(val_losses)), val_losses, label='validation loss')

        plt.savefig('loss_train_test.png')
        model.save_weights('s2s_attention.h5')

    else:
        optimizer = Adam(clipnorm=1.0)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['categorical_accuracy'])

        print(model.summary())
        model.load_weights('s2s_attention.h5')
        print("Loaded previous model...")
        history = model.fit(x=[encoder_input_data, decoder_input_data], y=[decoder_target_data],
                  batch_size=32,
                  epochs=400,
                  validation_split=0.1)

        plt.plot(range(len(history.history['loss'])), history.history['loss'], label='train loss')
        plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], label='validation loss')

        plt.savefig('loss_train_test.png')

        # Save model
        model.save_weights('s2s_attention.h5')


    def decode_sequence(input_seq):
        decoder_input = np.zeros(shape=(max_length, num_decoder_tokens))
        decoder_input[0, 128] = 1.
        for i in range(1, 600):
            #print(np.argmax(decoder_input, axis=-1)[:10])
            output = model.predict([input_seq, np.expand_dims(decoder_input, axis=0)]).argmax(axis=-1)
            output_char = output[0, i]
            #print("output_char: ", output_char)
            decoder_input[i, output_char] = 1.
        return np.argmax(decoder_input, axis=-1)

    ind = 13
    input_seq = np.expand_dims(encoder_input_data[ind], axis=0)
    res = decode_sequence(input_seq)
    print(res)
    # print(len(res))
    print(np.argmax(decoder_input_data, axis=-1)[ind])
    # print(np.argmax(decoder_target_data, axis=-1)[ind][:40])

    r = 5
    # for i in range(5, 30, 2):
    #     res2 = model.predict([np.expand_dims(encoder_input_data[ind][:i], axis=0),
    #                          np.expand_dims(decoder_input_data[ind][:i], axis=0)])
    #     print(np.argmax(res2, axis=-1))


if __name__ == "__main__":
    main()
    # get_data()
