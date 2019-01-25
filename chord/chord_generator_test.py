from chord.generator.chord_generator import ChordGenerator

def chord_to_id_test():
    cg = ChordGenerator('')
    assert cg.chord_to_id("A#:maj") == 22
    assert cg.chord_to_id("C:min") == 3
    assert cg.chord_to_id("Gb:maj(9)") == 14
    assert cg.chord_to_id("D:sus") == 6
    assert cg.chord_to_id("F:min(5)") == 13

def id_to_chord_test():
    cg = ChordGenerator('')
    assert cg.id_to_chord(22) == "A#:maj"
    assert cg.id_to_chord(3) == "C:min"
    assert cg.id_to_chord(14) == "F#:maj"
    assert cg.id_to_chord(6) == "D:maj"
    assert cg.id_to_chord(13) == "F:min"

def test_2():
    cg = ChordGenerator('')
    a = ['C:maj', 'C:min', 'C#:maj', 'C#:min', 'D:maj', 'D:min',
         'D#:maj', 'D#:min', 'E:maj', 'E:min', 'F:maj', 'F:min',
         'F#:maj', 'F#:min', 'G:maj', 'G:min', 'G#:maj', 'G#:min',
         'A:maj', 'A:min', 'A#:maj', 'A#:min', 'B:maj', 'B:min']
    for chord in a:
        print(cg.chord_to_id(chord), end=' ')

