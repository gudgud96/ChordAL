from collections import Counter

import seq2seq
from keras.optimizers import Adam
from keras.utils import to_categorical
from seq2seq.models import SimpleSeq2Seq

import numpy as np

from dataset.data_pipeline import DataPipeline


def get_data():
    dp = DataPipeline()
    chords, melodies = dp.get_csv_nottingham_cleaned()

    # get rid of leading and trailing zeros in melodies, and trim chords accordingly
    # for i in range(len(melodies)):
    #     pre_len = len(melodies[i])
    #     temp_melody = np.trim_zeros(melodies[i], 'f')
    #     len_change = pre_len - len(temp_melody)
    #     temp_chords = chords[i][len_change:]         # trim leading zeros
    #
    #     pre_len = len(temp_melody)
    #     temp_melody = np.trim_zeros(temp_melody, 'b')
    #     len_change = pre_len - len(temp_melody)
    #     temp_chords = temp_chords[:len(temp_chords) - len_change]
    #
    #     temp_melody = np.insert(temp_melody, 0, 128)
    #     temp_melody = np.insert(temp_melody, temp_melody.shape[-1], 129)
    #
    #     # padding to ensure the tensors have same length
    #     if len(temp_melody) < 600:
    #         melodies[i] = np.pad(temp_melody, (0, 600 - len(temp_melody)), mode='constant', constant_values=0)
    #     if len(temp_chords) < 600:
    #         chords[i] = np.pad(temp_chords, (0, 600 - len(temp_chords)), mode='constant', constant_values=0)

    melodies_target = np.copy(melodies)
    melodies_target = melodies_target[:, 1:]
    melodies_target = np.insert(melodies_target, melodies_target.shape[-1], 129, axis=-1)
    print(chords.shape, melodies.shape, melodies_target.shape)
    return chords, melodies, melodies_target


def main():
    encoder_input_data, decoder_input_data, decoder_target_data = get_data()
    encoder_input_data = to_categorical(encoder_input_data, num_classes=26)
    decoder_input_data = to_categorical(decoder_input_data, num_classes=130)
    decoder_target_data = to_categorical(decoder_target_data, num_classes=130)

    model = SimpleSeq2Seq(input_dim=26, hidden_dim=64, output_length=600, output_dim=130)
    optimizer = Adam(clipnorm=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    model.fit(encoder_input_data, decoder_input_data,
              batch_size=32, epochs=2)
    model.save_weights('s2s_simple.h5')

    res = model.predict(np.expand_dims(encoder_input_data[15], axis=0))
    print(np.argmax(res, axis=-1))


if __name__ == "__main__":
    main()