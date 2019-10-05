=========
Partitura
=========

Partitura is a Python 3 package for handling symbolic musical information. It
supports loading from and exporting to *MusicXML* and *MIDI* files.

The full documentation for `partitura` is available online at `readthedocs.org
<https://partitura.readthedocs.io/en/latest/index.html>`_.


Quickstart
==========

The following code loads the contents of an example MusicXML file included in
the package:

>>> import partitura
>>> my_xml_file = partitura.EXAMPLE_MUSICXML
>>> part = partitura.load_musicxml(my_xml_file)

The following shows the contents of the part:

>>> print(part.pretty())
Part id="P1" name="Piano"
 │
 ├─ TimePoint t=0 quarter=12
 │   │
 │   └─ starting objects
 │       │
 │       ├─ Measure number=1
 │       ├─ Note id=n01 voice=1 staff=2 type=whole pitch=A4
 │       ├─ Page number=1
 │       ├─ Rest id=r01 voice=2 staff=1 type=half
 │       ├─ System number=1
 │       └─ TimeSignature 4/4
 │
 ├─ TimePoint t=24 quarter=12
 │   │
 │   ├─ ending objects
 │   │   │
 │   │   └─ Rest id=r01 voice=2 staff=1 type=half
 │   │
 │   └─ starting objects
 │       │
 │       ├─ Note id=n02 voice=2 staff=1 type=half pitch=C5
 │       └─ Note id=n03 voice=2 staff=1 type=half pitch=E5
 │
 └─ TimePoint t=48 quarter=12
     │
     └─ ending objects
         │
         ├─ Measure number=1
         ├─ Note id=n01 voice=1 staff=2 type=whole pitch=A4
         ├─ Note id=n02 voice=2 staff=1 type=half pitch=C5
         └─ Note id=n03 voice=2 staff=1 type=half pitch=E5
  
The notes in this part can be accessed through the property
`part.notes`:

>>> part.notes
[<partitura.score.Note object at 0x...>, <partitura.score.Note object at 0x...>, 
 <partitura.score.Note object at 0x...>]

To create a piano roll extract from the part as a numpy array you would do
the following:

>>> import numpy as np
>>> pianoroll = np.array([(n.start.t, n.end.t, n.midi_pitch) for n in part.notes])
>>> print(pianoroll)
[[ 0 48 69]
 [24 48 72]
 [24 48 76]]

The note start and end times are in the units specificied by the
`divisions` element of the MusicXML file. This element specifies the
duration of a quarter note. The `divisions` value can vary within an
MusicXML file, so it is generally better to work with musical time in
beats.

The part object has a property :`part.beat_map` that converts timeline
times into beat times:

>>> beat_map = part.beat_map
>>> print(beat_map(pianoroll[:, 0]))
[0. 2. 2.]
>>> print(beat_map(pianoroll[:, 1]))
[4. 4. 4.]

More elaborate examples can be found in the documentation.

License
=======

The code in this package is licensed under the Apache 2.0 License. For details,
please see the `LICENSE <LICENSE>`_ file.

Installation
============

The easiest way to install the package is via ``pip`` from the `PyPI (Python
Package Index) <https://pypi.python.org/pypi>`_::

  pip install partitura

This will install the latest release of the package and will install all
dependencies automatically.

Mailing list
============

The `mailing list <https://groups.google.com/d/forum/partitura-users>`_ should be
used to get in touch with the developers and other users.

