'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Generate chord sequence given tempo slots.

Improvements needed:
( ) Play with Lakh dataset
( ) For chord extraction in scraped MIDI, the stream objects are Voices instead of Measures. If so, 
    how long the voice should we take for chord extraction?

'''

from music21 import *
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('test.log', mode='w')
handler.setLevel(logging.INFO)
# add the handlers to the logger
logger.addHandler(handler)

# to parse a song and understand the structure of its music21 stream object
def parse(a, string, num):
    if num > 1000:
        return
    if hasattr(a, '__len__'):
        logger.info(string + str(a) + ' ' + str(len(a)))
        for i in range(len(a)):
            parse(a[i], string + '--', num + 1)
    else:
        if type(a) == note.Note:
            logger.info(string +  str(a))
        else:
            logger.info(string + str(a))

a = converter.parse('Hello Muddah, Hello Fadduh (Allan Sherman).mid')
parse(a, '', 0)

measures = a.getElementsByClass(stream.Part)[0].getElementsByClass(stream.Voice)[0].getElementsByClass(note.Note)
for note in measures:
    print(note)
for measure in a.getElementsByClass(stream.Measure):
    print(measure)

a.show()
