import os

import keras
import tensorflow as tf
import numpy as np

from note.chord_to_note_generator import ChordToNoteGenerator
from keras import backend as K
import pandas as pd
from pandas import DataFrame
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():

    ROOT_DIR = './graphs/'
    os.makedirs(ROOT_DIR, exist_ok=True)
    OUTPUT_MODEL_FILE_NAME = os.path.join(ROOT_DIR,'tf.ckpt')

    # get the keras model
    ctng = ChordToNoteGenerator()
    ctng.load_model(model_name="bidem", is_fast_load=True)
    model = ctng.model
    layer = model.layers[0] # embedding layer

    inp = model.input  # input placeholder
    output = layer.output  # embedding layer outputs
    functor = K.function([inp, K.learning_phase()], [output])   # evaluation functions

    # Testing
    test = np.arange(2, 25)
    test = np.pad(test, (0, 1200 - len(test)), 'constant', constant_values=0)
    layer_outs = np.array(functor([test])).squeeze()
    layer_outs = layer_outs[:24]

    # Get working directory
    PATH = os.getcwd()

    # Path to save the embedding and checkpoints generated
    LOG_DIR = PATH + '/graphs/'

    # Load data
    df = pd.DataFrame(layer_outs)
    df.to_csv('test.csv')

    # Load the metadata file. Metadata consists your labels. This is optional.
    # Metadata helps us visualize(color) different clusters that form t-SNE
    # metadata = os.path.join(LOG_DIR, 'df_labels.tsv')

    # Generating PCA and
    pca = PCA(n_components=32,
              random_state=123,
              svd_solver='full'
              )
    df_pca = pd.DataFrame(pca.fit_transform(df))
    df_pca = df_pca.values

    # TensorFlow Variable from data
    tf_data = tf.Variable(df_pca)

    # Running TensorFlow Session
    with tf.Session() as sess:
        saver = tf.train.Saver([tf_data])
        sess.run(tf_data.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'tf_data.ckpt'))
        config = projector.ProjectorConfig()

        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = tf_data.name

        # Link this tensor to its metadata(Labels) file
        # embedding.metadata_path = metadata

        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)


if __name__ == "__main__":
    main()