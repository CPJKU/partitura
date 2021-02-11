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

To load the score in python we first import the partitura package:

>>> import partitura

For convenience a MusicXML file with the above contents is included in the
package. The path to the file is stored as :const:`partitura.EXAMPLE_MUSICXML`, so
that we load the above score as follows:

>>> my_musicxml_file = partitura.EXAMPLE_MUSICXML
>>> part = partitura.load_musicxml(my_musicxml_file)


Displaying the typeset part
===========================

The :func:`partitura.render` function displays the part as a typeset score:

>>> partitura.render(part)

.. image:: images/score_example.png
   :alt: Score example
   :align: center
        
This should open an image of the score in the default image viewing
application of your desktop. The function requires that either `MuseScore 
<https://musescore.org/>`_ or `lilypond <http://lilypond.org/>`_ is
installed on your computer.


Exporting a score to MusicXML
=============================

The :func:`partitura.save_musicxml` function exports score information to
MusicXML. The following line saves `part` to a file `mypart.musicxml`:

>>> partitura.save_musicxml(part, 'mypart.musicxml')


Viewing the musical elements
============================

The function :func:`~partitura.load_musicxml` returns the score as a
:class:`~partitura.score.Part` instance. When we print it, it displays its
id and part-name:

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

`structured numpy array <https://numpy.org/doc/stable/user/basics.rec.html>`_

.. doctest::

  >>> arr = part.note_array
  >>> arr.dtype # doctest: +NORMALIZE_WHITESPACE
  dtype([('onset_beat', '<f4'), ('duration_beat', '<f4'),
         ('onset_quarter', '<f4'), ('duration_quarter', '<f4'),
		 ('onset_div', '<i4'), ('duration_div', '<i4'),
		 ('pitch', '<i4'), ('voice', '<i4'), ('id', '<U256')])

>>> for pitch, onset, duration in zip(arr["pitch"], arr["onset_beat"], arr["duration_beat"]):
...     print(pitch, onset, duration)
69 0.0 4.0
72 2.0 2.0
76 2.0 2.0

The note start and end times are in the units specificied by the
`divisions` element of the MusicXML file. This element specifies the
duration of a quarter note. The `divisions` value can vary within an
MusicXML file, so it is generally better to work with musical time in
beats.

The part object has a property :attr:`part.beat_map
<partitura.score.Part.beat_map>` that converts timeline times into beat
times:

..
    >>> beat_map = part.beat_map
    >>> print(beat_map(pianoroll[:, 0]))
    [0. 2. 2.]
    >>> print(beat_map(pianoroll[:, 1]))
    [4. 4. 4.]


Iterating over arbitrary musical objects
========================================

In the previous Section we used :attr:`part.notes
<partitura.score.Part.notes>` to obtain the notes in the part as a list.
This property is a short cut for the following statement:

.. doctest::

  >>> list(part.iter_all(partitura.score.Note)) # doctest: +NORMALIZE_WHITESPACE
  [<partitura.score.Note object at 0x...>, <partitura.score.Note object at 0x...>, 
  <partitura.score.Note object at 0x...>]

Here we access the :meth:`~partitura.score.Part.iter_all` method. Given a
class, it iterates over all instances of that class that occur in the part:

>>> for m in part.iter_all(partitura.score.Measure):
...     print(m)
0--48 Measure number=1

The :meth:`~partitura.score.Part.iter_all` method has a keyword
`include_subclasses` that indicates that we are also interested in any
subclasses of the specified class. For example, the following statement
iterates over all objects in the part:

>>> for m in part.iter_all(object, include_subclasses=True):
...     print(m)
0--48 Note id=n01 voice=1 staff=2 type=whole pitch=A4
0--24 Rest id=r01 voice=2 staff=1 type=half
0--48 Page number=1
0--48 System number=1
0--48 Measure number=1
0-- TimeSignature 4/4
24--48 Note id=n02 voice=2 staff=1 type=half pitch=C5
24--48 Note id=n03 voice=2 staff=1 type=half pitch=E5


This approach is useful for example when we want to retrieve rests in
addition to notes. Since rests and notes are both subclassess of
:class:`GenericNote <partitura.score.GenericNote>`, the following works:

>>> for m in part.iter_all(partitura.score.GenericNote, include_subclasses=True):
...     print(m)
0--48 Note id=n01 voice=1 staff=2 type=whole pitch=A4
0--24 Rest id=r01 voice=2 staff=1 type=half
24--48 Note id=n02 voice=2 staff=1 type=half pitch=C5
24--48 Note id=n03 voice=2 staff=1 type=half pitch=E5

By default, `include_subclasses` is False.

..
   Importing data from MIDI
   ========================

   >>> part = partitura.load_midi()


Creating a musical score by hand
================================

You can build a musical score from scratch, by creating a `Part` object. We
start by renaming the `partitura.score` module to `score`, for convenience:

>>> import partitura.score as score

Then we create an empty part with id 'P0' and name 'My Part' (the name is
optional, the id is mandatory), and a quarter note
duration of 10 units.

>>> part = score.Part('P0', 'My Part', quarter_duration=10)

Adding elements to the part is done by the
:meth:`~partitura.score.Part.add` method, which takes a musical element,
a start and an end time. Either of the `start` and `end` arguments can be
omitted, but if both are omitted the method will do nothing.

We now add a 3/4 time signature at t=0, and three notes. The notes are
instantiated by specifying an (optional) id, pitch information, and an
(optional) voice:

>>> part.add(score.TimeSignature(3, 4), start=0)
>>> part.add(score.Note(id='n0', step='A', octave=4, voice=1), start=0, end=10)
>>> part.add(score.Note(id='n1', step='C', octave=5, alter=1, voice=2), start=0, end=10)
>>> part.add(score.Note(id='n2', step='C', octave=5, alter=1, voice=2), start=10, end=40)

Note that the duration of notes is not hard-coded in the Note instances, but
defined implicitly by their start and end times in the part.

Here's what the part looks like:

>>> print(part.pretty())
Part id="P0" name="My Part"
 │
 ├─ TimePoint t=0 quarter=10
 │   │
 │   └─ starting objects
 │       │
 │       ├─ 0--10 Note id=n0 voice=1 staff=None type=quarter pitch=A4
 │       ├─ 0--10 Note id=n1 voice=2 staff=None type=quarter pitch=C#5
 │       └─ 0-- TimeSignature 3/4
 │
 ├─ TimePoint t=10 quarter=10
 │   │
 │   ├─ ending objects
 │   │   │
 │   │   ├─ 0--10 Note id=n0 voice=1 staff=None type=quarter pitch=A4
 │   │   └─ 0--10 Note id=n1 voice=2 staff=None type=quarter pitch=C#5
 │   │
 │   └─ starting objects
 │       │
 │       └─ 10--40 Note id=n2 voice=2 staff=None type=half. pitch=C#5
 │
 └─ TimePoint t=40 quarter=10
     │
     └─ ending objects
         │
         └─ 10--40 Note id=n2 voice=2 staff=None type=half. pitch=C#5

We see that the notes n0, n1, and n2 have been correctly recognized as
quarter, quarter, and dotted half, respectively.

Let's save the part to MusicXML:

>>> partitura.save_musicxml(part, 'mypart.musicxml')

When we look at the contents of `mypart.musicxml`, surprisinly, the `<part></part>` element is empty:

.. code-block:: xml

    <?xml version='1.0' encoding='UTF-8'?>
    <!DOCTYPE score-partwise PUBLIC
      "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
      "http://www.musicxml.org/dtds/partwise.dtd">
    <score-partwise>
      <part-list>
        <score-part id="P0">
          <part-name>My Part</part-name>
        </score-part>
      </part-list>
      <part id="P0"/>
    </score-partwise>

The problem with our newly created part is that it contains no
measures. Since the MusicXML format requires musical elements to be
contained in measures, saving the part to MusicXML omits the objects we
added.


Adding measures
===============

One option to add measures is to add them by hand like we've added the
notes and time signature. A more convenient alternative is to use the
function :func:`~partitura.score.add_measures`:
 
>>> score.add_measures(part)

This function uses the time signature information in the part to add
measures accordingly:

>>> print(part.pretty())
Part id="P0" name="My Part"
 │
 ├─ TimePoint t=0 quarter=10
 │   │
 │   └─ starting objects
 │       │
 │       ├─ 0--30 Measure number=1
 │       ├─ 0--10 Note id=n0 voice=1 staff=None type=quarter pitch=A4
 │       ├─ 0--10 Note id=n1 voice=2 staff=None type=quarter pitch=C#5
 │       └─ 0-- TimeSignature 3/4
 │
 ├─ TimePoint t=10 quarter=10
 │   │
 │   ├─ ending objects
 │   │   │
 │   │   ├─ 0--10 Note id=n0 voice=1 staff=None type=quarter pitch=A4
 │   │   └─ 0--10 Note id=n1 voice=2 staff=None type=quarter pitch=C#5
 │   │
 │   └─ starting objects
 │       │
 │       └─ 10--40 Note id=n2 voice=2 staff=None type=half. pitch=C#5
 │
 ├─ TimePoint t=30 quarter=10
 │   │
 │   ├─ ending objects
 │   │   │
 │   │   └─ 0--30 Measure number=1
 │   │
 │   └─ starting objects
 │       │
 │       └─ 30--40 Measure number=2
 │
 └─ TimePoint t=40 quarter=10
     │
     └─ ending objects
         │
         ├─ 30--40 Measure number=2
         └─ 10--40 Note id=n2 voice=2 staff=None type=half. pitch=C#5

Let's see what our part with measures looks like in typeset form:
         
>>> partitura.render(part)

.. image:: images/score_example_1.png
   :alt: Part with measures
   :align: center

Although the notes are there, the music is not typeset correctly, since the
first measure should have a duration of three quarter notes, but instead is
has a duration of four quarter notes. The problem is that the note *n2*
crosses a measure boundary, and thus should be tied.

Splitting up notes using ties
=============================

In musical notation notes that span measure boundaries are split up, and then
tied together. This can be done automatically using the function
:func:`~partitura.score.tie_notes`:

>>> score.tie_notes(part)
>>> partitura.render(part)

.. image:: images/score_example_2.png
   :alt: Part with measures
   :align: center

Now the score looks correct. Displaying the contents reveals that the part
now has an extra quarter note *n2a* that starts at the measure boundary,
whereas the note *n2* is now a half note, ending at the measure boundary.

>>> print(part.pretty())
Part id="P0" name="My Part"
 │
 ├─ TimePoint t=0 quarter=10
 │   │
 │   └─ starting objects
 │       │
 │       ├─ 0--30 Measure number=1
 │       ├─ 0--10 Note id=n0 voice=1 staff=None type=quarter pitch=A4
 │       ├─ 0--10 Note id=n1 voice=2 staff=None type=quarter pitch=C#5
 │       └─ 0-- TimeSignature 3/4
 │
 ├─ TimePoint t=10 quarter=10
 │   │
 │   ├─ ending objects
 │   │   │
 │   │   ├─ 0--10 Note id=n0 voice=1 staff=None type=quarter pitch=A4
 │   │   └─ 0--10 Note id=n1 voice=2 staff=None type=quarter pitch=C#5
 │   │
 │   └─ starting objects
 │       │
 │       └─ 10--30 Note id=n2 voice=2 staff=None type=half tie_group=n2+n2a pitch=C#5
 │
 ├─ TimePoint t=30 quarter=10
 │   │
 │   ├─ ending objects
 │   │   │
 │   │   ├─ 0--30 Measure number=1
 │   │   └─ 10--30 Note id=n2 voice=2 staff=None type=half tie_group=n2+n2a pitch=C#5
 │   │
 │   └─ starting objects
 │       │
 │       ├─ 30--40 Measure number=2
 │       └─ 30--40 Note id=n2a voice=2 staff=None type=quarter tie_group=n2+n2a pitch=C#5
 │
 └─ TimePoint t=40 quarter=10
     │
     └─ ending objects
         │
         ├─ 30--40 Measure number=2
         └─ 30--40 Note id=n2a voice=2 staff=None type=quarter tie_group=n2+n2a pitch=C#5


Removing elements
=================

Just like we can add elements to a part, we can also remove them, using the
:meth:`~partitura.score.Part.remove` method. The following lines remove the
measure instances that were added using the
:func:`~partitura.score.add_measures` function:

>>> for measure in list(part.iter_all(score.Measure)):
...     part.remove(measure)

Note that we create a list of all measures in `part` before we remove them. This is necessary to avoid changing the contents of `part` while we iterate over it.


Music Analysis
==============

The package offers tools for various types music analysis, including key estimation, tonal tension estimation, voice separation, and pitch spelling. The functions take the note information of in the form of a  `structured numpy array <https://numpy.org/doc/stable/user/basics.rec.html>`_, as returned by the :attr:`~partitura.score.Part.note_array` attribute.

Key Estimation
--------------

Key estimation is performed by the function
:func:`~partitura.musicanalysis.estimate_key`. The function returns a string representation of the root and mode of the key:

>>> key_name = partitura.musicanalysis.estimate_key(part.note_array)
>>> print(key_name)
C#m

The number of sharps/flats and the mode can be inferred from the key name using the convenience function :func:`~partitura.utils.key_name_to_fifths_mode`:

>>> partitura.utils.key_name_to_fifths_mode(key_name)
(4, 'minor')



Loading MIDI files
==================

Partitura contains three functions to load MIDI files: 
:func:`~partitura.midi_to_notearray`:
:func:`~partitura.load_performance_midi`:
:func:`~partitura.load_score_midi`:

The first and easiest function is :func:`~partitura.midi_to_notearray`:. It loads a MIDI file into 
a structured array of MIDI notes given by onset and duration (in seconds), pitch, velocity, and ID. 

>>> my_midi_file = partitura.EXAMPLE_MIDI
>>> midi_note_array = partitura.midi_to_notearray(my_midi_file)
>>> midi_note_array

The note_array is a structured numpy array, individual fields can be accessed using the field names as strings, 
e.g. midi_note_array["onset_sec"]

The second function is :func:`~partitura.midi_to_notearray`:. This function returns a :class:`~partitura.score.Measure`