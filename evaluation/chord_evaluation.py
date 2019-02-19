import os

from chord.chord_generator import ChordGenerator, CHORD_DICT, CHORD_SEQUENCE_FILE

# Determine whether to test on major chords, minor chords, or all
IS_MAJOR = False
IS_MINOR = True


def evaluate_chord_function(chords):
    cg = ChordGenerator()
    chord_length = len(chords)
    tonic_key, tonic_tonality = chords[0].split(':')
    k = CHORD_DICT[tonic_key]
    function_dict = initialize_function_dict(k, tonic_tonality)
    chord_functions = []
    outlier = 0
    non_function_chord = 0
    expected_function = ["tonic"]
    last_chord = ""
    chord_repeats = 0

    for i in range(len(chords)):
        chord = chords[i]
        chord_id = cg.chord_to_id(chord)

        # find repeating chords
        if last_chord == chords[i]:
            chord_repeats += 1
        last_chord = chords[i]

        # find outlier
        if chord_id not in function_dict:
            outlier += 1
            chord_functions.append('x')

        # find non-functioning chords as defined by Riemann
        else:
            chord_func = function_dict[chord_id]
            if len(chord_func) == 1:
                if expected_function and chord_func[0] not in expected_function:
                    non_function_chord += 1
                    chord_functions.append('--' + chord_func[0] + '--')
                else:
                    chord_functions.append(chord_func[0])
            else:   # must have tonic
                if expected_function:
                    func = set(chord_func).intersection(set(expected_function))
                    if "tonic" in func:
                        chord_functions.append("tonic")
                    else:
                        chord_functions.append(list(func)[0])
                else:   # when the previous term is an outlier
                    chord_functions.append("tonic")

        expected_function = get_expected_function(chord_functions[-1])

    result = {}
    result["outlier"] = outlier / chord_length
    result["non_function"] = non_function_chord / chord_length
    result["chord_repeats"] = chord_repeats / chord_length
    print(result)

    return result


def evaluate_chord_variation_coefficient(chords, result):
    '''
    Simple evaluations on chords generated.
    :param chords:
        Chords sequence generated.
    :return:
        No return, but 3 metrics printed -
            Chord variation - count dict of the chord sequence
            Chord variation coefficient - 1 - (# unique chords / # total chords)
            Chord distribution variance - Variance of counts for each chord
    '''
    cg = ChordGenerator()
    chords = [cg.chord_to_id(chord) for chord in chords]
    chords = [cg.id_to_chord(id) for id in chords]
    count_dict = {}
    for chord in chords:
        if chord not in count_dict:
            count_dict[chord] = chords.count(chord)

    variation_coefficient = 1 - len(list(count_dict.keys())) / len(chords)
    result["variation"] = variation_coefficient

    return result


def get_expected_function(chord_function):
    if chord_function == "tonic":
        return ["tonic", "subdominant", "dominant"]
    elif chord_function == "subdominant":
        return ["subdominant", "dominant"]
    elif chord_function == "dominant":
        return ["dominant", "tonic"]


def initialize_function_dict(k, tonic_tonality):
    result_dict = {}

    if tonic_tonality == "maj":
        result_dict[2 * k] = ["tonic"]
        result_dict[2 * round(k + 2) + 1] = ["subdominant"]
        result_dict[2 * round(k + 4) + 1] = ["tonic", "dominant"]
        result_dict[2 * round(k + 5)] = ["subdominant"]
        result_dict[2 * round(k + 7)] = ["dominant"]
        result_dict[2 * round(k + 9) + 1] = ["tonic", "subdominant"]
        result_dict[2 * round(k + 11) + 1] = ["dominant"]

    else:
        result_dict[2 * k + 1] = ["tonic"]
        result_dict[2 * round(k + 2) + 1] = ["subdominant"]
        result_dict[2 * round(k + 4)] = ["tonic", "dominant"]
        result_dict[2 * round(k + 5) + 1] = ["subdominant"]
        result_dict[2 * round(k + 7)] = ["dominant"]
        result_dict[2 * round(k + 9)] = ["tonic", "subdominant"]
        result_dict[2 * round(k + 11) + 1] = ["dominant"]

    return result_dict


def round(note_value):
    note_value = note_value % 12
    if note_value == 0:
        note_value = 12
    return note_value


def generate_chord_samples():
    cg = ChordGenerator(CHORD_SEQUENCE_FILE)

    if IS_MAJOR:
        sample_chords = open("sample_chords_major.txt").readlines()
    elif IS_MINOR:
        sample_chords = open("sample_chords_minor.txt").readlines()
    else:
        sample_chords = open("sample_chords.txt").readlines()

    sample_chords = [chord.strip() for chord in sample_chords]
    chords_experiment = 'chords_experiment.txt'
    f = open(chords_experiment, 'a+')

    for chord in sample_chords:
        chord = chord.split(' > ')
        generated_chords = cg.generate_chords(chord.copy(), num_of_chords=16)
        f.write(' > '.join(generated_chords) + '\n')
        generated_chords = cg.generate_chords(chord.copy(), num_of_chords=16)
        f.write(' > '.join(generated_chords) + '\n')
        generated_chords = cg.generate_chords(chord.copy(), num_of_chords=32)
        f.write(' > '.join(generated_chords) + '\n')
        generated_chords = cg.generate_chords(chord.copy(), num_of_chords=32)
        f.write(' > '.join(generated_chords) + '\n')


def move_files_to_evaluation_folder():
    folder_name = sorted([int(k) for k in os.listdir('./evaluation_results')])[-1] + 1
    os.mkdir('./evaluation_results/' + str(folder_name) + '/')
    os.rename('evaluation.txt', 'evaluation_results/' + str(folder_name) + '/evaluation.txt')
    os.rename('chords_experiment.txt', 'evaluation_results/' + str(folder_name) + '/chords_experiment.txt')
    os.remove('example_chord.txt')
    os.remove('result_chord.txt')


def main():
    chords_experiment = 'chords_experiment.txt'
    evaluation_text = 'evaluation.txt'
    open(chords_experiment, 'w+').write('')
    open(evaluation_text, 'w+').write('')

    generate_chord_samples()

    f = [k.lstrip().rstrip() for k in open(chords_experiment).readlines()]
    e = open(evaluation_text, 'a+')

    total_dict = {}
    for i in range(len(f)):
        line = f[i]
        if line == '':
            continue
        chords = line.split(' > ')
        result = evaluate_chord_function(chords)
        result = evaluate_chord_variation_coefficient(chords, result)
        e.write(str(chords) + '\n')

        temp = []
        for k in result.keys():
            if k not in total_dict:
                total_dict[k] = 0
            e.write("{}  -  {}\n".format(k, result[k]))
            total_dict[k] += result[k]
            temp.append(result[k])

        e.write('\n\n')

    e.write('-- Average --\n')
    temp = []
    for k in total_dict.keys():
        e.write('{}  -  {}\n'.format(k, total_dict[k] / len(f)))
        temp.append(total_dict[k] / len(f))

    e.close()
    move_files_to_evaluation_folder()


if __name__ == "__main__":
    main()

