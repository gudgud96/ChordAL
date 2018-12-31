import datetime
import random

from app import app
from flask import render_template, request, flash
from generator.song_generator import generate_song
import os

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
@app.route('/index')
def index():
    songs = os.listdir('app/static')
    chosen_songs = random.sample(songs, 5)
    answer_file = open('temp_answer.txt', 'w+')

    for i in range(len(chosen_songs)):
        if 'song' in chosen_songs[i]:
            s = "MIDIjs.play('static/{}');".format(chosen_songs[i])
            answer_file.write('static/{}'.format(chosen_songs[i]) + ',0\n')           # human - 0, AI - 1
        else:
            s = "MIDIjs.play('static/{}/song.mid');".format(chosen_songs[i])
            answer_file.write('static/{}/song.mid'.format(chosen_songs[i]) + ',1\n')
        chosen_songs[i] = s

    answer_file.close()
    return render_template('index.html', song1=chosen_songs[0], song2=chosen_songs[1],
                           song3=chosen_songs[2], song4=chosen_songs[3], song5=chosen_songs[4])

@app.route("/play", methods=['POST'])
def play():
    # Get chords and bar numbers from user
    # flash('Loading...')
    chords = request.form['chords']
    bar_number = request.form['bar_number']

    # Preprocess chords and bar numbers
    if chords == '':
        if bar_number == '':
            generate_song()
        else:
            generate_song(bar_number=int(bar_number))
    else:
        chord_lst = chords.replace(' ', '').split(',')
        for i in range(len(chord_lst)):
            key, tonality = chord_lst[i][:-1], chord_lst[i][-1]
            tonality = 'maj' if tonality == '+' else 'min'
            key = key.upper()
            chord_lst[i] = key + ':' + tonality

        if bar_number == '':
            generate_song(chords=chord_lst)
        else:
            generate_song(chords=chord_lst, bar_number=int(bar_number))

    # Move generated song files
    folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.mkdir('app/static/' + folder_name + '/')
    os.rename('melody.mid', 'app/static/' + folder_name + '/melody.mid')
    os.rename('chords.mid', 'app/static/' + folder_name + '/chords.mid')
    os.rename('song.mid', 'app/static/' + folder_name + '/song.mid')
    os.rename('example_chord.txt', 'app/static/' + folder_name + '/example_chord.txt')
    os.rename('result_chord.txt', 'app/static/' + folder_name + '/result_chord.txt')

    message = "MIDIjs.play('static/" + folder_name + "/song.mid');"
    return render_template('play.html', message=message)


@app.route("/turing", methods=['POST'])
def turing_test():
    answers = open('temp_answer.txt').readlines()
    songs = [k.lstrip().rstrip().split(',')[0] for k in answers]
    answers = [k.lstrip().rstrip().split(',')[1] for k in answers]
    user = [request.form['r' + str(k)] for k in range(1, 6)]
    ai_as_human, human_as_ai = 0, 0

    response_db = open('responses.txt', 'a+')
    for i in range(len(answers)):
        if answers[i] == '0' and user[i] == '1':
            human_as_ai += 1
        elif answers[i] == '1' and user[i] == '0':
            ai_as_human += 1

        response_db.write(songs[i] + ',' + user[i] + '\n')
        answers[i] = 'Human' if answers[i] == "0" else 'AI'
        user[i] = 'Human' if user[i] == "0" else 'AI'

    os.remove('temp_answer.txt')
    songs = ["MIDIjs.play('{}');".format(k) for k in songs]
    return render_template('turing_eval.html', song1=songs[0], song2=songs[1],
                           song3=songs[2], song4=songs[3], song5=songs[4], song1_user=user[0],
                           song2_user=user[1], song3_user=user[2], song4_user=user[3],
                           song5_user=user[4], song1_ans=answers[0], song2_ans=answers[1],
                           song3_ans=answers[2], song4_ans=answers[3], song5_ans=answers[4],
                           human_as_ai=human_as_ai, ai_as_human=ai_as_human)