============
Introduction
============

The principal aim of the `partitura` package is to handle richly structured
musical information as conveyed by modern staff music notation. It provides
a much wider range of possibilities to deal with music than the more
reductive (but very common) pianoroll-oriented approach inspired by the
MIDI standard.

Specifically, the package allows for representing a variety of information
in musical scores beyond the onset, duration and MIDI pitch numbers of
notes, such as:

* pitch spellings,
* symbolic duration categories,
* and voicing information.

Moreover, it supports musical notions that are not note-related, like:

* measures,
* tempo indications,
* performance directions,
* repeat structures,
* and time/key signatures.

In addition to score information, the package can load MIDI recordings of
performed scores, and alignments between scores and performances in the
`Matchfile` format used by the publicly available `Vienna4x22 piano corpus
research dataset
<https://repo.mdw.ac.at/projects/IWK/the_vienna_4x22_piano_corpus/data/index.html>`_.

Supported file types
====================

Musical data can be loaded from and saved to `MusicXML` and `MIDI`
files. Furthermore, `partitura` uses `MuseScore <https://musescore.org/>`_
as a backend to load files in other formats, like `MuseScore`, `MuseData`,
and `GuitarPro`. This requires a working installation of MuseScore on your
computer.


Conceptual Overview
===================

This section offers some conceptual and design considerations that may be
helpful when working with the package.

Representing score information
------------------------------

The package defines a musical ontology to describe musical
scores that roughly follows the elements defined by the `MusicXML
specification <http://usermanuals.musicxml.com/MusicXML/MusicXML.htm>`_.
More specifically, the elements of a musical score are represented as a
collection of instances of classes like `Note`, `Measure`, `Slur`, and
`Rest`. These instances are attached to an instance of class `Part`, which
corresponds to the role of an instrument in a musical score. A part may
contain one or more staffs, depending on the instrument.

In contrast to MusicXML documents, where musical time is largely implicit,
time plays a crucial role in the representation of scores in
`partitura`. Musical elements are associated to a `Part` instance by
specifying their *start* (and possibly *end*) times. The `Part` instance
thus acts as a timeline consisting of a number of discrete timepoints, each
of which holds references to the musical elements starting and ending at
that time. The musical elements themselves contain references to their
respective starting and ending timepoints. Other than that,
cross-references between musical elements are used sparingly, to keep the
API simple.

Musical elements in a `Part` can be filtered by class and iterated over,
either from a particular timepoint onward or backward, or within a
specified range. For example to find the measure to which a note belongs,
you would iterate backwards over elements of class Measure that start at or
before the start time of the note and select the first element of that
iteration.


Score vs. performance
---------------------

Although the MIDI format can be used to represent both score-related
(key/time signatures, tempo) and performance-related information
(expressive timing, dynamics), partitura regards a MIDI file as a
representation of either a a score or a performance. Therefore is has
separate functions to load and save scores
(:func:`~partitura.load_score_midi`, :func:`~partitura.save_score_midi`)
and performances (:func:`~partitura.load_performance_midi`,
:func:`~partitura.save_performance_midi`). :func:`~partitura.load_score_midi`
offers simple quantization for unquantized MIDIs but in general you should
not expect a MIDI representation of a performance to be loaded correctly as
a `Part` instance.


Relation to `music21 <https://web.mit.edu/music21/>`_
=====================================================

The `music21` package has been around since 2008, and is one of the few
python packages available for working with symbolic musical data. It is
both more mature and more elaborate than `partitura`.  The aims of
`partitura` are different from and more modest than those of `music21`,
which aims to provide a toolkit for computer-aided musicology. Instead,
`partitura` intends to provide a convenient way to work with symbolic
musical data in the context of problems such as musical expression
modeling, or music generation.  Although it is not the main aim of the
package to provide music analysis tools, the package does offer
functionality for pitch spelling, voice assignment and key estimation.

