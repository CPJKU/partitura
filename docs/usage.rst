=====
Usage
=====

In this Section we will demonstrate basic usage of the package.

Importing MusicXML
==================

We start by loading a score from a MusicXML file. As an example we take a
MusicXML file with the following contents:

.. literalinclude:: ../partitura/assets/score_example.musicxml
   :language: xml

The typeset score looks like this:

.. image:: images/score_example.png
   :alt: Score example
   :align: center
        
To load the score in python we first import the partitura package:

>>> import partitura

For convenience a MusicXML file with the above contents is included in the
package. The path to the file is stored as :const:`partitura.EXAMPLE_MUSICXML`, so
that we load the above score as follows:

>>> my_musicxml_file = partitura.EXAMPLE_MUSICXML
>>> part = partitura.load_musicxml(my_musicxml_file)

Viewing the musical elements
============================

The function :func:`~partitura.load_musicxml` returns the score as a :class:`~partitura.score.Part` instance. When we
print it, it displays its id and part-name:

>>> print(part)
Part id="P1" name="Piano"

To see all of the elements in the part at once, we can call its
:meth:`~partitura.score.Part.pretty` method:

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

This reveals that the part has three time points at which one or more musical
objects start or end. At `t=0` there are several starting objects, including a
:class:`~partitura.score.TimeSignature`, :class:`~partitura.score.Measure`,
:class:`~partitura.score.Page`, and :class:`~partitura.score.System`.

Extracting a piano roll
=======================

The notes in this part can be accessed through the property
:attr:`part.notes <partitura.score.Part.notes>`:

.. doctest::

  >>> part.notes # doctest: +NORMALIZE_WHITESPACE
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

The part object has a property :attr:`part.beat_map
<partitura.score.Part.beat_map>` that converts timeline times into beat
times:

>>> beat_map = part.beat_map
>>> print(beat_map(pianoroll[:, 0]))
[0. 2. 2.]
>>> print(beat_map(pianoroll[:, 1]))
[4. 4. 4.]

Iterating over arbitrary musical objects
========================================

In the previous Section we used :attr:`part.notes <partitura.score.Part.notes>` to obtain the notes in the part as a list.
This property is a short cut for the following statement:

.. doctest::

  >>> list(part.iter_all(partitura.score.Note)) # doctest: +NORMALIZE_WHITESPACE
  [<partitura.score.Note object at 0x...>, <partitura.score.Note object at 0x...>, 
  <partitura.score.Note object at 0x...>]

Here we access the :meth:`~partitura.score.Part.iter_all` method. Given a class,
it iterates over all instances of that class that occur in the part:

>>> for m in part.iter_all(partitura.score.Measure):
...     print(m)
Measure number=1

The :meth:`~partitura.score.Part.iter_all` method has a keyword
`include_subclasses` that indicates that we are also interested in any
subclasses of the specified class. For example, the following statement
iterates over all objects in the part:

>>> for m in part.iter_all(object, include_subclasses=True):
...     print(m)
Page number=1
System number=1
Measure number=1
TimeSignature 4/4
Note id=n01 voice=1 staff=2 type=whole pitch=A4
Rest id=r01 voice=2 staff=1 type=half
Note id=n02 voice=2 staff=1 type=half pitch=C5
Note id=n03 voice=2 staff=1 type=half pitch=E5

This approach is useful for example when we want to retrieve rests in
addition to notes. Since rests and notes are both subclassess of
:class:`GenericNote <partitura.score.GenericNote>`, the following works:

>>> for m in part.iter_all(partitura.score.GenericNote, include_subclasses=True):
...     print(m)
Note id=n01 voice=1 staff=2 type=whole pitch=A4
Rest id=r01 voice=2 staff=1 type=half
Note id=n02 voice=2 staff=1 type=half pitch=C5
Note id=n03 voice=2 staff=1 type=half pitch=E5

By default, `include_subclasses` is False.

..
   Importing data from MIDI
   ========================

   >>> part = partitura.load_midi()


Creating a score by hand
========================

You can build a score from scratch, by creating a `Part` object:

>>> import partitura.score as score

>>> part = score.Part('My Part')
>>> part.set_quarter_duration(0, 10)
>>> ts = score.TimeSignature(3, 4)
>>> note1 = score.Note(step='A', octave=4) # A4
>>> note2 = score.Note(step='C', octave=5, alter=1) # C#5

and adding them to the part:

>>> part.add(ts, 0)
>>> part.add(note1, 0, 15)
>>> part.add(note2, 0, 20)
>>> score.add_measures(part)

Exporting a score to MusicXML
=============================

The :func:`partitura.save_musicxml` function exports score information to
MusicXML. The following statement saves `part` to a file `mypart.musicxml`:

>>> partitura.save_musicxml(part, 'mypart.musicxml')

