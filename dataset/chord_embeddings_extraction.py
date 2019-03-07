import numpy as np
from tqdm import tqdm

from chord.chord_generator import ChordGenerator
from keras import backend as K
import pickle


def extract_embeddings_to_dict():
    cg = ChordGenerator()
    model = cg.build_model()
    model.load_weights('../chord/chord_weights_bidem.h5')
    layer = model.layers[0] # embedding layer

    inp = model.input  # input placeholder
    output = layer.output  # embedding layer outputs
    functor = K.function([inp, K.learning_phase()], [output])   # evaluation functions

    # Testing
    test = np.arange(2, 26)
    
    # test = np.pad(test, (0, 1200 - len(test)), 'constant', constant_values=0)
    layer_outs = np.array(functor([test])).squeeze()

    print(layer_outs)
    chord_embeddings_dict = {}

    for i in range(len(layer_outs)):
        chord_embeddings_dict[i + 2] = layer_outs[i]

    pickle_out = open("chord_embeddings_dict.pickle", "wb")
    pickle.dump(chord_embeddings_dict, pickle_out)
    pickle_out.close()


def convert_dataset_to_embeddings():
    pickle_in = open("chord_embeddings_dict.pickle", "rb")
    embeddings_dict = pickle.load(pickle_in)
    embeddings_dict[0] = np.zeros((32,))

    chords = np.load('csv-nottingham/chords_merged_cleaned.npy')
    new_chords = []
    cur_chord = []
    for i in tqdm(range(len(chords))):
        for j in range(len(chords[i])):
            cur_chord.append(embeddings_dict[chords[i][j]])
        new_chords.append(cur_chord)
        cur_chord = []

    new_chords = np.array(new_chords)
    print(new_chords.shape)
    np.save("csv-nottingham/chords_merged_cleaned_embeddings.npy", new_chords)


if __name__ == "__main__":
    convert_dataset_to_embeddings()
    # demonstration
