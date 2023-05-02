[//]: # (<p align="center"> )

[//]: # (<img src="./partitura/assets/partitura_logo_final.jpg" height="200">)

[//]: # (</p>)
<p align="center">
    <img src="partitura/assets/partitura_logo_black.png#gh-light-mode-only" height="200">
    <img align="center" src="./partitura/assets/partitura_logo_white.png#gh-dark-mode-only" height="200">
</p>

[![Latest Release](https://img.shields.io/github/v/release/cpjku/partitura)](https://github.com/cpjku/partitura/releases)
[![Pypi Package](https://badge.fury.io/py/partitura.svg)](https://badge.fury.io/py/partitura)
[![Unittest Status](https://github.com/CPJKU/partitura/workflows/Partitura%20Unittests/badge.svg)](https://github.com/CPJKU/partitura/actions?query=workflow%3A%22Partitura+Unittests%22)
[![CodeCov Status](https://codecov.io/gh/CPJKU/partitura/branch/develop/graph/badge.svg?token=mnZ234sGSA)](https://codecov.io/gh/CPJKU/partitura)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)



**Partitura** is a Python package for handling symbolic musical information. It
supports loading from and exporting to *MusicXML* and *MIDI* files. It also supports loading from _Humdrum *kern*_ and *MEI*.

The full documentation for `partitura` is available online at [readthedocs.org](https://partitura.readthedocs.io/en/latest/index.html).

User Installation
==========

The easiest way to install the package is via `pip` from the [PyPI (Python
Package Index)](https://pypi.python.org/pypi>):
```shell
pip install partitura
```
This will install the latest release of the package and will install all dependencies automatically.


Quickstart
==========
A detailed tutorial with some hands-on MIR applications is available [here](https://cpjku.github.io/partitura_tutorial/index.html).

The following code loads the contents of an example MusicXML file included in
the package:
```python
import partitura as pt
my_xml_file = pt.EXAMPLE_MUSICXML
score = pt.load_score(my_xml_file)
```
The partitura `load_score` function will import any score format, i.e. (Musicxml, Kern, MIDI or MEI) to a `partitura.Score` object.
The score object will contain all the information in the score, including the score parts.
The following shows the contents of the first part of the score:

```python
part = score.parts[0]
print(part.pretty())
```
Output:
```shell
Part id="P1" name="Piano"
 │
 ├─ TimePoint t=0 quarter=12
 │   │
 │   └─ starting objects
 │       │
 │       ├─ 0--48 Measure number=1
 │       ├─ 0--48 Note id=n01 voice=1 staff=2 type=whole pitch=A4
 │       ├─ 0--48 Page number=1
 │       ├─ 0--24 Rest id=r01 voice=2 staff=1 type=half
 │       ├─ 0--48 System number=1
 │       └─ 0-- TimeSignature 4/4
 │
 ├─ TimePoint t=24 quarter=12
 │   │
 │   ├─ ending objects
 │   │   │
 │   │   └─ 0--24 Rest id=r01 voice=2 staff=1 type=half
 │   │
 │   └─ starting objects
 │       │
 │       ├─ 24--48 Note id=n02 voice=2 staff=1 type=half pitch=C5
 │       └─ 24--48 Note id=n03 voice=2 staff=1 type=half pitch=E5
 │
 └─ TimePoint t=48 quarter=12
     │
     └─ ending objects
         │
         ├─ 0--48 Measure number=1
         ├─ 0--48 Note id=n01 voice=1 staff=2 type=whole pitch=A4
         ├─ 24--48 Note id=n02 voice=2 staff=1 type=half pitch=C5
         ├─ 24--48 Note id=n03 voice=2 staff=1 type=half pitch=E5
         ├─ 0--48 Page number=1
         └─ 0--48 System number=1
  
```
If `lilypond` or `MuseScore` are installed on the system, the following command
renders the part to an image and displays it:

```python
pt.render(part)
```
![Score example](https://raw.githubusercontent.com/CPJKU/partitura/main/docs/source/images/score_example.png)


The notes in this part can be accessed through the property
`part.notes`:

```python
part.notes
> [<partitura.score.Note object at 0x...>, <partitura.score.Note object at 0x...>, 
> <partitura.score.Note object at 0x...>]

```


The following code stores the start, end, and midi pitch of the notes in a numpy
array:

```python
import numpy as np
pianoroll = np.array([(n.start.t, n.end.t, n.midi_pitch) for n in part.notes])
print(pianoroll)
> [[ 0 48 69]
>  [24 48 72]
>  [24 48 76]]
```


The note start and end times are in the units specified by the
`divisions` element of the MusicXML file. This element specifies the
duration of a quarter note. The `divisions` value can vary within an
MusicXML file, so it is generally better to work with musical time in
beats.

The part object has a property :`part.beat_map` that converts timeline
times into beat times:

```python
beat_map = part.beat_map
print(beat_map(pianoroll[:, 0]))
> [0. 2. 2.]
print(beat_map(pianoroll[:, 1]))
> [4. 4. 4.]
```


The following commands save the part to MIDI and MusicXML, or export it as a WAV file (using [additive synthesis](https://en.wikipedia.org/wiki/Additive_synthesis)), respectively:

```python
# Save Score MIDI to file.
pt.save_score_midi(part, 'mypart.mid')

# Save Score MusicXML to file.
pt.save_musicxml(part, 'mypart.musicxml')

# Save as audio file using additive synthesis
pt.save_wav(part, 'mypart.wav')
```


More elaborate examples can be found in the `documentation
<https://partitura.readthedocs.io/en/latest/index.html>`_.

Import other formats
====================
For **MusicXML** files do:

```python
import partitura as pt
my_xml_file = pt.EXAMPLE_MUSICXML
score = pt.load_musicxml(my_xml_file)
```

For **Kern** files do:

```python
import partitura as pt
my_kern_file = pt.EXAMPLE_KERN
score = pt.load_kern(my_kern_file)
```

For **MEI** files do:

```python
import partitura as pt
my_mei_file = pt.EXAMPLE_MEI
score = pt.load_mei(my_mei_file)
```


One can also import any of the above formats by just using:

```python
import partitura as pt
any_score_format_path = pt.EXAMPLE_MUSICXML
score = pt.load_score(any_score_format_path)
```


License
=======

The code in this package is licensed under the Apache 2.0 License. For details,
please see the [LICENSE](LICENSE) file.


Citing Partitura
================

If you find Partitura useful, we would appreciate if you could cite us!

```
@inproceedings{partitura_mec,
  title={{Partitura: A Python Package for Symbolic Music Processing}},
  author={Cancino-Chac\'{o}n, Carlos Eduardo and Peter, Silvan David and Karystinaios, Emmanouil and Foscarin, Francesco and Grachten, Maarten and Widmer, Gerhard},
  booktitle={{Proceedings of the Music Encoding Conference (MEC2022)}},
  address={Halifax, Canada},
  year={2022}
}
```

[//]: # (    | `Grachten, M. <https://maarten.grachten.eu>`__, `Cancino-Chacón, C. <http://www.carloscancinochacon.com>`__ and `Gadermaier, T. <https://www.jku.at/en/institute-of-computational-perception/about-us/people/thassilo-gadermaier/>`__)

[//]: # (    | "`partitura: A Python Package for Handling Symbolic Musical Data <http://carloscancinochacon.com/documents/extended_abstracts/GrachtenEtAl-ISMIR2019-LBD-ext-abstract.pdf>`__\ ".)

[//]: # (    | Late Breaking/Demo Session at the 20th International Society for)

[//]: # (    Music Information Retrieval Conference, Delft, The Netherlands,)

[//]: # (    2019.)



Acknowledgments
===============

This project receives funding from the European Research Council (ERC) under 
the European Union's Horizon 2020 research and innovation programme under grant 
agreement No 101019375 ["Whither Music?"](https://www.jku.at/en/institute-of-computational-perception/research/projects/whither-music/).



This work has received support from the European Research Council (ERC) under
the European Union’s Horizon 2020 research and innovation programme under grant
agreement No. 670035 project ["Con Espressione"](https://www.jku.at/en/institute-of-computational-perception/research/projects/con-espressione/)
and the Austrian Science Fund (FWF) under grant P 29840-G26 (project
["Computer-assisted Analysis of Herbert von Karajan's Musical Conducting Style"](https://karajan-research.org/programs/musical-interpretation-karajan))
<p align="center">
    <img src="docs/source/images/aknowledge_logo.png#gh-light-mode-only" height="200">
    <img src="docs/source/images/aknowledge_logo_negative.png#gh-dark-mode-only" height="200">
</p>

[//]: # ()
[//]: # (.. image:: https://raw.githubusercontent.com/CPJKU/partitura/master/docs/images/erc_fwf_logos.jpg)

[//]: # (   :width: 600 px)

[//]: # (   :scale: 1%)

[//]: # (   :align: center)
