## ChordAL - A chord-based approach for AI music generation
![](logo-small.png)

Extended abstract: TBA
<br>
Full thesis: TBA

[![SoundCloud](https://jisungk.github.io/deepjazz/img/button_soundcloud.png)](https://soundcloud.com/hord-hord-basedomposer)  
Check out ChordAL's music on [Soundcloud](https://soundcloud.com/hord-hord-basedomposer)!

### Introduction
It seems to be very intuitive for human composers to write songs based on chord progressions. However, not much work is done on exploring
how we could generate melodies based on given chord progressions. 

Here we propose **ChordAL**, which is a chord-based learning system for music composition using deep learning.

### Architecture
The generation process is divided into 3 parts: *chord generation*, *chord-to-note generation*, and *music styling*.
For more information, please refer to our [extended abstract]().

We see chord-to-note generation tasks to be similar as **neural machine translation** tasks, which
trains on a large amount of *parallel datasets* between chords and notes.


### File Structure
Each folder is structured to be a component of ChordAL's system:
 - `chord`: chord generation
 - `dataset`: data-loading related functions
 - `evaluation`: automated evaluation test on generated piece
 - `generator`: combines chord and melody with music styling
 - `models`: models used / experimented in ChordAL
 - `note`: melody generation based on chords
 - `visualizer`: Flask web application for demonstration
 
### Dependencies
- [Keras](https://github.com/keras-team/keras)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [music21](https://github.com/cuthbertLab/music21)
- [pretty_midi](https://github.com/craffel/pretty-midi)
- [mido](https://github.com/mido/mido)

### Try ChordAL yourself!
```bibtex
pip install -r requirements.txt
cd visualizer
python3 main.py
``` 
You can then try it on your browser at `localhost:5000`. Enjoy and hope you like it!

### References
For training, we use [McGill Chord Dataset](http://ddmal.music.mcgill.ca/research/salami/annotations), 
[Nottingham dataset](https://github.com/jukedeck/nottingham-dataset) cleaned by Jukedeck, 
and [CSV leadsheet database](http://marg.snu.ac.kr/chord_generation/) published by MARG Seoul, Korea.
These are the more-established parallel dataset of chords and notes, to the best
of our knowledge.

Some code references from [Keras](https://github.com/keras-team/keras)
and [Tensorflow](https://github.com/tensorflow/tensorflow) tutorials are being used during development.

### Author
[Tan Hao Hao](https://gudgud96.github.io)
<br>
Nanyang Technological University, School of Computer Science and Engineering
<br>
tanh0207 (at) e.ntu.edu.sg

Code is licensed under the MIT License.