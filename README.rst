=========
Partitura
=========

Partitura is a Python package for handling symbolic musical information. It
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
  
If `lilypond` or `MuseScore` are installed on the system, the following command
renders the part to an image and displays it:

>>> partitura.render(part)

.. image:: https://raw.githubusercontent.com/CPJKU/partitura/master/docs/images/score_example.png
   :alt: Score example

The notes in this part can be accessed through the property
`part.notes`:

>>> part.notes
[<partitura.score.Note object at 0x...>, <partitura.score.Note object at 0x...>, 
 <partitura.score.Note object at 0x...>]

The following code stores the start, end, and midi pitch of the notes in a numpy
array:

>>> import numpy as np
>>> pianoroll = np.array([(n.start.t, n.end.t, n.midi_pitch) for n in part.notes])
>>> print(pianoroll)
[[ 0 48 69]
 [24 48 72]
 [24 48 76]]

The note start and end times are in the units specified by the
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

The following commands save the part to MIDI and MusicXML, respectively:

>>> partitura.save_score_midi(part, 'mypart.mid')

>>> partitura.save_musicxml(part, 'mypart.musicxml')

More elaborate examples can be found in the `documentation
<https://partitura.readthedocs.io/en/latest/index.html>`_.


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

Citing Partitura
================

If you find Partitura useful, we would appreciate if you could cite us!

    | `Grachten, M. <https://maarten.grachten.eu>`__, `Cancino-Chacón, C. <http://www.carloscancinochacon.com>`__ and `Gadermaier, T. <https://www.jku.at/en/institute-of-computational-perception/about-us/people/thassilo-gadermaier/>`__
    | "`partitura: A Python Package for Handling Symbolic Musical Data <http://carloscancinochacon.com/documents/extended_abstracts/GrachtenEtAl-ISMIR2019-LBD-ext-abstract.pdf>`__\ ".
    | Late Breaking/Demo Session at the 20th International Society for
    Music Information Retrieval Conference, Delft, The Netherlands,
    2019.



Acknowledgments
===============

This work has received support from the European Research Council (ERC) under
the European Union’s Horizon 2020 research and innovation programme under grant
agreement No. 670035 (project `"Con Espressione"
<https://www.jku.at/en/institute-of-computational-perception/research/projects/con-espressione/>`_)
and the Austrian Science Fund (FWF) under grant P 29840-G26 (project
`"Computer-assisted Analysis of Herbert von Karajan's Musical Conducting Style"
<https://karajan-research.org/programs/musical-interpretation-karajan>`_)

.. image:: https://raw.githubusercontent.com/CPJKU/partitura/master/docs/images/erc_fwf_logos.jpg
   :width: 600 px
   :scale: 1%
   :align: center
