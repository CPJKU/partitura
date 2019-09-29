=========
Partitura
=========

Partitura is a Python 3 package for handling symbolic musical information. It
supports loading from and exporting to *MusicXML* and *MIDI* files.

Quickstart
==========

The following code loads one or more parts from a MusicXML file `score.musicxml`
and shows their contents:
  
  >>> import partitura
  >>> import partitura.score as score
  
  >>> parts = partitura.load_musicxml("score.musicxml")
  >>> for part in score.iter_parts(parts):
  >>>     print(part.pretty())
    
For a given *Part* object `part`, the following code extracts the note onsets,
offsets, and MIDI pitches:

  >>> notes = part.get_all(score.Note)
  >>> pianoroll = [(n.start.t, n.end.t, n.midi_pitch) for n in notes]

Documentation
=============

The documentation for `partitura` is available online at `readthedocs.org
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

Mailing list
============

The `mailing list <https://groups.google.com/d/forum/partitura-users>`_ should be
used to get in touch with the developers and other users.

