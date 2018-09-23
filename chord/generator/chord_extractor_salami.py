'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Extract chord sequence from McGill Billboard annotations as dataset.

Improvements needed:
( ) The dataset only has 891 sequences (isn't it 1300?). We should find more.
( ) Pair it up with genre or mood data.
( ) Now we only take major and minor. Should cover others (blues, sevenths, sus, etc.)
( ) There may be 2 chords in one bar. For now we split it and treat a different chord at a different bar.
    We should carefully consider this again.

'''
import os
import re
from tqdm import tqdm

chord_text_list = []
root = '../dataset/McGill-Billboard/'
for path, subdirs, files in os.walk(root):
    for name in files:
        chord_text_list.append(os.path.join(path, name).replace('\\', '/'))

chord_sequence_file = open('chord_sequence_file.txt', 'w+')

index = 230
sequence_check = ''

for i in tqdm(range(1, len(chord_text_list))):    # 1st file is an index file
    string = open(chord_text_list[i]).read()
    b = re.findall('\|\s[A-Z].+\s\|', string)
    b = ' | '.join([b1[2:-2] for b1 in b]).split(' | ')
    new_b = []
    for key in b:
        # There may be 2 chords in one bar. For now we split it and treat a different chord at a different bar.
        keys = key.split(' ')       
        for k in keys:
            m = re.match('[ABCDEFG][#b]?:[(maj)|(min)].+', k)   # only take major and minor here. can expand
            if m != None:
                new_b.append(m.group(0))
    sequence = ' > '.join(new_b)
    if i == index:
        sequence_check = sequence
    chord_sequence_file.write(sequence + '\n')

# Just for checking
assert sequence_check == open('chord_sequence_file.txt', 'r').readlines()[index - 1].lstrip().rstrip()
