from chord.chord_generator import CHORD_DICT, DECODE_DICT

def main():
    f = open('chord_sequence_all_no_repeat.txt').readlines()
    f = [k.lstrip().rstrip() for k in f]
    for j in range(len(f)):
        sequence = f[j]
        chords = sequence.split(' > ')
        first_chord = chords[0]
        first_key, first_tonality = first_chord.split(':')
        first_key_index = CHORD_DICT[first_key]
        key_of_c = 1
        diff = first_key_index - key_of_c
        if diff > 0:
            for i in range(len(chords)):
                chord = chords[i]
                key, tonality = chord.split(':')
                key_index = CHORD_DICT[key]
                transposed_key_index = key_index - diff
                if transposed_key_index <= 0:
                    transposed_key_index += 12
                transposed_key = DECODE_DICT[transposed_key_index]
                chords[i] = ':'.join([transposed_key, tonality])
        f[j] = ' > '.join(chords)

    new_f = open('chord_sequence_all_no_repeat_normalized.txt', 'w+')
    for line in f:
        new_f.write(line + '\n')


if __name__ == "__main__":
    main()