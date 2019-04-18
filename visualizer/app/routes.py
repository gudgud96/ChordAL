import datetime

from app import app
from flask import render_template, request, flash
from generator.song_generator import generate_song, song_styling
import os

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
MODEL_DICT = {"Bidirectional": "bidem", "Regularized": "bidem_regularized",
              "Embedding": "bidem_preload", "Seq2seq": "seq2seq"}


@app.route('/')
@app.route('/index')
def index():
    return render_template('test_generate.html')


@app.route("/play", methods=['POST'])
def play():
    # Get chords and bar numbers from user
    print("in play...")
    chords = request.form['chords']
    model_name = "bidem_preload"            # fix the model to bidirectional embeddings for now
    style = request.form['style'].lower()
    bar_number = 32 if '32' in request.form['bar_number'] else 16
    if style == '':
        style = 'strings'

    # Preprocess chords and bar numbers
    if chords == '':
        generate_song(style=style, model_name=model_name, bar_number=bar_number)

    else:
        chord_lst = chords.replace(' ', '').split(',')
        for i in range(len(chord_lst)):
            key, tonality = chord_lst[i][:-1], chord_lst[i][-1]
            tonality = 'maj' if tonality == '+' else 'min'
            key = key.upper()
            if key[-1] == "B" and len(key) == 2:
                key = key[:-1] + "b"        # don't make flat sign uppercase
            chord_lst[i] = key + ':' + tonality
        generate_song(chords=chord_lst, style=style, model_name=model_name, bar_number=bar_number)

    # Move generated song files
    folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.mkdir('app/static/' + folder_name + '/')
    os.rename('melody.mid', 'app/static/' + folder_name + '/melody.mid')
    os.rename('chords.mid', 'app/static/' + folder_name + '/chords.mid')
    os.rename('song.mid', 'app/static/' + folder_name + '/song.mid')
    # os.rename('song.wav', 'app/static/' + folder_name + '/song.wav')
    os.rename('example_chord.txt', 'app/static/' + folder_name + '/example_chord.txt')
    os.rename('result_chord.txt', 'app/static/' + folder_name + '/result_chord.txt')

    message = "MIDIjs.play('static/" + folder_name + "/song.mid');"
    download_link = 'static/' + folder_name + "/song.mid"

    return render_template('test_play.html', message=message,
                           style=None, download_link=download_link)


@app.route("/style", methods=['POST'])
def style():
    style = request.form['style']
    songs = os.listdir('app/static')
    songs.remove('human-song')
    songs.remove('styles')
    song = songs[-1]
    melody_file = 'app/static/{}/melody.mid'.format(song)
    chord_file = 'app/static/{}/chords.mid'.format(song)
    song_file = 'app/static/{}/song-{}.mid'.format(song, style.lower())
    song_styling(melody_file, chord_file, song_file, style=style.lower())

    songs = os.listdir('app/static')
    songs.remove('human-song')
    songs.remove('styles')
    song = songs[-1]
    # print(songs)
    message = "MIDIjs.play('static/" + song + "/song-{}.mid');".format(style.lower())
    download_link = 'static/' + song + '/song-{}.mid'.format(style.lower())

    return render_template('test_play.html', message=message,
                           style=style, download_link=download_link)