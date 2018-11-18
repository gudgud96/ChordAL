'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Model Builder - should have this component that stores all kinds of candidate models.

Improvements needed:
( ) - For VAE, try to bring custom loss with KL divergence in
'''
from keras import objectives
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Lambda
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt


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

        plt.plot(range(len(history.history['loss'])), history.history['loss'], label='train loss')
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

    def train_model(self, model, epochs, loss='mean_squared_error'):
        print(model.summary())
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        history = model.fit(self.X_train, self.X_train, epochs=epochs)

        scores = model.evaluate(self.X_train, self.X_train, verbose=True)
        print('Train loss:', scores[0])
        print('Train accuracy:', scores[1])
        scores = model.evaluate(self.X_test, self.X_test, verbose=True)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        plt.plot(range(len(history.history['loss'])), history.history['loss'], label='train loss')
        plt.show()

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