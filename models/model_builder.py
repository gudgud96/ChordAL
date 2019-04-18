'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Model Builder - should have this component that stores all kinds of candidate models.

Improvements needed:
(/) - For VAE, try to bring custom loss with KL divergence in
'''
from keras import objectives
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Lambda, Dropout, TimeDistributed, Activation, Conv2D, MaxPooling2D, \
    Flatten, Convolution2D, GRU, LeakyReLU, CuDNNGRU, Embedding, Bidirectional, dot, concatenate, CuDNNLSTM
from keras.regularizers import l1, l2
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

from chord.chord_generator import NUM_CLASSES
from models.keras_attention_wrapper import AttentionDecoder
from keras.optimizers import Adam


class ModelBuilder:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def build_seq2seq_model(self, num_encoder_tokens, num_decoder_tokens, latent_dim):

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
        return model

    def build_stacked_autoencoder(self, input_dim, intermediate_dim):
        '''
        Build a stacked autoencoder.
        :param input_dim: input dimension
        :param intermediate_dim: eg. [512,256,128,256,512]
        :return: model
        '''
        model = Sequential()
        for i in range(len(intermediate_dim)):
            if i == 0:
                model.add(Dense(units=intermediate_dim[i],
                                activation='relu', input_shape=(input_dim,)))
            else:
                model.add(Dense(units=intermediate_dim[i],
                                activation='intermediate_dim'))
        model.add(Dense(input_dim))
        return model

    def build_and_train_vae(self, input_dim, intermediate_dim, latent_dim,
                                      epochs=200, batch_size=128):

        # build vae, encoder and decoder
        vae, encoder, generator, z_log_var, z_mean = self.build_vae_model(input_dim,
                                                                          intermediate_dim, latent_dim)
        vae.compile(optimizer='adam', loss=vae_loss_custom(z_log_var, z_mean), metrics=['accuracy'])
        history = vae.fit(self.X_train, self.X_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.X_test, self.X_test))

        t(range(len(history.history['loss'])), history.history['loss'], label='train loss')
        plt.show()

        return vae, encoder, generator

    def build_vae_model(self, input_dim, intermediate_dim, latent_dim):
        inputs = Input(shape=(input_dim,), name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # random sample
        z = Lambda(sampling, name='z')([z_mean, z_log_var])

        # build decoder model
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(input_dim)
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        print('input_dim', input_dim)
        print(x_decoded_mean.shape)

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # instantiate decoder model (or also known as generator)
        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)

        # instantiate VAE model
        vae = Model(inputs, x_decoded_mean, name='vae_mlp')

        return vae, encoder, generator, z_log_var, z_mean

    def build_basic_rnn_model(self, input_dim, output_dim=None, use_dropout=False):
        '''
        Build basic RNN model using LSTMs.
        :param input_dim: input dimension, normally (100, 128 * 12)
        :param use_dropout: whether to use dropout
        :return: model
        '''
        if not output_dim:
            output_dim = input_dim[-1]
        model = Sequential()
        # added reset after flag for CuDNN compatibility purpose
        model.add(CuDNNLSTM(64, return_sequences=True, input_shape=input_dim))
        model.add(CuDNNLSTM(128, return_sequences=True))
        model.add(Dropout(0.8))
        # model.add(TimeDistributed(Dense(input_dim[-2] * input_dim[-3])))
        model.add(TimeDistributed(Dense(output_dim)))
        model.add(Activation('softmax'))
        return model

    def build_bidirectional_rnn_model_no_embeddings(self, input_dim, output_dim=128):
        model = Sequential()
        model.add(Bidirectional(LSTM(64, bias_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), return_sequences=True), input_shape=input_dim))
        model.add(Dropout(0.4))
        model.add(Bidirectional(LSTM(64, bias_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), return_sequences=True)))
        model.add(Dropout(0.4))
        model.add(TimeDistributed(Dense(output_dim)))  # 128 notes to output, multi-class
        model.add(Activation('softmax'))
        return model

    def build_bidirectional_rnn_model(self, input_dim, output_dim=128):
        '''
        Build bidirectional RNN model using LSTMs.
        :param input_dim: input dimension, normally (100, 128 * 12)
        :return: model
        '''
        model = Sequential()
        model.add(Embedding(NUM_CLASSES, 32, input_shape=input_dim))     # NUM_CLASSES is the total number of chord IDs
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_dim))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(output_dim)))                  # 128 notes to output, multi-class
        model.add(Activation('softmax'))
        return model

    def build_attention_bidirectional_rnn_model(self, input_dim):
        '''
        Build attention bidirectional RNN model using LSTMs.
        :param input_dim: input dimension, normally (100, 128 * 12)
        :return: model
        '''
        encoder_input = Input(shape=input_dim)

        encoder = Embedding(NUM_CLASSES, 32, input_shape=input_dim)(encoder_input)
        encoder = Bidirectional(LSTM(64, return_sequences=True))(encoder)
        encoder = Dropout(0.2)(encoder)
        encoder = Bidirectional(LSTM(128, return_sequences=True))(encoder)
        encoder = Dropout(0.2)(encoder)

        decoder = Bidirectional(LSTM(128, return_sequences=True))(encoder)
        attention = dot([decoder, encoder], axes=[2, 2])
        attention = Activation('softmax', name='attention')(attention)
        print('attention', attention)

        context = dot([attention, encoder], axes=[2, 1])
        print('context', context)

        decoder_combined_context = concatenate([context, decoder])
        print('decoder_combined_context', decoder_combined_context)

        # Has another weight + tanh layer as described in equation (5) of the paper
        # output = TimeDistributed(Dense(64, activation="tanh"))(decoder_combined_context)
        output = TimeDistributed(Dense(128, activation="softmax"))(decoder_combined_context)
        print('output', output)
        print('decoder', decoder)

        model = Model(inputs=[encoder_input], outputs=[output])
        return model

    def build_basic_conv2d_rnn_model(self, input_dim, use_dropout=False):
        '''
        Build basic Conv2d -> RNN model using LSTMs.
        :param input_dim: input dimension, normally (100, 128 * 12)
        :param use_dropout: whether to use dropout
        :return: model
        '''
        print(input_dim)
        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), padding='same'), input_shape=input_dim))
        model.add(TimeDistributed(MaxPooling2D()))
        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dense(32)))

        model.add(GRU(64, return_sequences=True, input_shape=input_dim))
        model.add(LeakyReLU(alpha=0.3))
        # model.add(LSTM(64, return_sequences=True))
        if use_dropout:
            model.add(Dropout(0.8))
        model.add(TimeDistributed(Dense(input_dim[-2] * input_dim[-3])))
        # model.add(TimeDistributed(Dense(input_dim[-1])))
        model.add(Activation('sigmoid'))
        return model

    def build_basic_cnn_model(self, input_dim, use_dropout=False):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same'), input_shape=input_dim)
        model.add(MaxPooling2D())
        model.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(TimeDistributed(Dense(32)))

    def train_model(self, model, epochs, loss='mean_squared_error'):
        print(model.summary())
        loss_metrics_dict = {
            "mean_squared_error": ['accuracy'],
            "binary_crossentropy": ['binary_accuracy'],
            "categorical_crossentropy": ['categorical_accuracy']
        }
        optimizer = Adam(clipnorm=1.0)
        model.compile(loss=loss, optimizer=optimizer, metrics=loss_metrics_dict[loss])
        history = model.fit(self.X_train, self.Y_train, epochs=epochs, validation_data=(self.X_test, self.Y_test))

        scores = model.evaluate(self.X_train, self.Y_train, verbose=True)
        print('Train loss:', scores[0])
        print('Train accuracy:', scores[1])
        scores_2 = model.evaluate(self.X_test, self.Y_test, verbose=True)
        print('Test loss:', scores_2[0])
        print('Test accuracy:', scores_2[1])

        plt.plot(range(len(history.history['loss'])), history.history['loss'], label='train loss')
        plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], label='validation loss')

        plt.savefig('loss_train_test.png')
        open('train_test_accuracy.txt', 'w+').write(
            'Train acc: {} Test acc: {} Train_loss: {} Test_loss: {}'.format(scores[1],
                                                                             scores_2[1],
                                                                             scores[0],
                                                                             scores_2[0]))

        return model

    def train_with_generator(self, model, epochs, loss='mean_squared_error'):

        # generate embeddings and one-hot on the fly
        from utils import convert_chord_indices_to_embeddings

        def generate_training_data():
            while 1:
                for i in range(int(len(self.X_train) * 0.9), len(self.X_train)):
                    input_chord, output_note = self.X_test[i], self.Y_test[i]
                    input_chord = np.array(convert_chord_indices_to_embeddings(input_chord))
                    output_note = to_categorical(output_note, num_classes=128)
                    yield (np.expand_dims(input_chord, axis=0),
                           np.expand_dims(output_note, axis=0))

        def generate_validation_data():
            while 1:
                for i in range(int(len(self.X_train) * 0.9), len(self.X_train)):
                    input_chord, output_note = self.X_test[i], self.Y_test[i]
                    input_chord = np.array(convert_chord_indices_to_embeddings(input_chord))
                    output_note = to_categorical(output_note, num_classes=128)
                    yield (np.expand_dims(input_chord, axis=0),
                           np.expand_dims(output_note, axis=0))

        print("Train with generator...")
        print(model.summary())
        loss_metrics_dict = {
            "mean_squared_error": ['accuracy'],
            "binary_crossentropy": ['binary_accuracy'],
            "categorical_crossentropy": ['categorical_accuracy']
        }
        optimizer = Adam(clipnorm=1.0)
        model.compile(loss=loss, optimizer=optimizer, metrics=loss_metrics_dict[loss])
        history = model.fit_generator(generate_training_data(),
                                      validation_data=generate_validation_data(),
                                      validation_steps=1,
                                      steps_per_epoch=1458, epochs=epochs)
        scores = model.evaluate(self.X_train, self.Y_train, verbose=True)
        print('Train loss:', scores[0])
        print('Train accuracy:', scores[1])
        scores_2 = model.evaluate(self.X_test, self.Y_test, verbose=True)
        print('Test loss:', scores_2[0])
        print('Test accuracy:', scores_2[1])

        plt.plot(range(len(history.history['loss'])), history.history['loss'], label='train loss')
        plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], label='validation loss')

        plt.savefig('loss_train_test.png')
        open('train_test_accuracy.txt', 'w+').write(
            'Train acc: {} Test acc: {} Train_loss: {} Test_loss: {}'.format(scores[1],
                                                                             scores_2[1],
                                                                             scores[0],
                                                                             scores_2[0]))

        return model


def sampling(args):
    """
    Reparameterization trick by sampling fr an isotropic unit Gaussian.
    instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def vae_loss_custom(z_log_var, z_mean):
    def vae_loss(x, x_decoded_mean):
        print('shape', x.shape, x_decoded_mean.shape)
        mse_loss = objectives.mean_squared_error(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return mse_loss + kl_loss
    return vae_loss

if __name__ == "__main__":
    a = ModelBuilder()
