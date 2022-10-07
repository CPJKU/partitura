============
Introduction
============
Partitura is a lightweight Python package for handling the musical information contained in symbolic music formats,
such as musical scores and MIDI performances. The package is built for researchers in the music information research (MIR) field
that need easy access to a large amount of musical information.

As opposed to audio files, symbolically encoded music
contains explicit note information, and organizes them notes in temporal and organizational structures such as measures, beats, parts, and voices.
It can also explicitly represent dynamics and temporal directives and other high-level musical
features such as time signature, pitch spelling, and key signatures.
While this rich set of musical elements adds useful information that can be leveraged by 
systems, it also drastically increases the complexity of encoding and processing symbolic musical
formats. Common formats for storage such as MEI, MusicXML, Humdrum \*\*kern and MIDI
are not ideally suited to be directly used as input in MIR tasks. Therefore, the typical data
processing pipeline starts with parsing the relevant information from those files and putting it
into a convenient data structure.

Partitura provides easy access to features commonly used in music information retrieval tasks, such as:

*  note arrays : lists of timed pitched events
*  pianorolls : 2D time x pitch matrices

It also support other score elements such
as time and key signatures, performance directives, and repeat structures. 

.. The principal aim of the `partitura` package is to handle richly structured
.. musical information as conveyed by modern staff music notation. It provides
.. a much wider range of possibilities to deal with music than the more
.. reductive (but very common) pianoroll-oriented approach inspired by the
.. MIDI standard.

.. Specifically, the package allows for representing a variety of information
.. in musical scores beyond the onset, duration and MIDI pitch numbers of
.. notes, such as:

.. * pitch spellings,
.. * symbolic duration categories,
.. * and voicing information.

.. Moreover, it supports musical notions that are not note-related, like:

.. * measures,
.. * tempo indications,
.. * performance directions,
.. * repeat structures,
.. * and time/key signatures.

.. In addition to handling score information, the package can load MIDI recordings of
.. performed scores, and alignments between scores and performances.

Supported file types
====================

Partitura can load musical scores (in MEI, MusicXML, Humdrum \*\*kern, and MIDI formats) 
and MIDI performances.

Furthermore, `partitura` uses `MuseScore <https://musescore.org/>`_
as a backend to load files in other formats, like `MuseScore`, `MuseData`,
and `GuitarPro`. This requires a working installation of MuseScore on your
computer.

Score-performance alignments can be read from different file types by
`partitura`.  Firstly, it supports reading from the `Matchfile` format used by
the publicly available `Vienna4x22 piano corpus research dataset
<https://repo.mdw.ac.at/projects/IWK/the_vienna_4x22_piano_corpus/data/index.html>`_.
Secondly there is read support for `Match` and `Corresp` files produced by
Nakamura's `music alignment software
<https://midialignment.github.io/demo.html>`_.


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
both more mature and more elaborate than `partitura` for tasks like creating
and manipulating score information and we suggest using it if
you are working in computational musicology. 

`Partitura` is instead built specifically for people that wants to apply machine 
learning and deep learning techniques to symbolic music data. Its focus is mainly 
on the extraction of relevant features from symbolic music data, in a fast way 
that require a minimal musical knowledge.
Moreover partitura supports MIDI performances and score-to-performances 
alignments, that are not handled by music21.

.. A hybrid music21 and partitura usage is also possible thanks to the music21 import function.
.. For example, you can load a score in music21, modify it, and then use the music21 to partitura converter
.. to get the score features that can be computed by partitura.

.. `partitura` are different from and more modest than those of `music21`,
.. which aims to provide a toolkit for computer-aided musicology. Instead,
.. `partitura` intends to provide a convenient way to work with symbolic
.. musical data in the context of problems such as musical expression modeling, or music generation.  Although it is not the main aim of the package to provide music analysis tools, the package does offer functionality for pitch spelling, voice assignment and key estimation.

Credits
=======

Citing Partitura
----------------

If you find Partitura useful, we would appreciate if you could cite us!


>>> @inproceedings{partitura_mec,
  title={{Partitura: A Python Package for Symbolic Music Processing}},
  author={Cancino-Chac\'{o}n, Carlos Eduardo and Peter, Silvan David and Karystinaios, Emmanouil and Foscarin, Francesco and Grachten, Maarten and Widmer, Gerhard},
  booktitle={{Proceedings of the Music Encoding Conference (MEC2022)}},
  address={Halifax, Canada},
  year={2022}
}


Acknowledgments
---------------

This project receives funding from the European Research Council (ERC) under
the European Union's Horizon 2020 research and innovation programme under grant
agreement No 101019375 `"Whither Music?" <https://www.jku.at/en/institute-of-computational-perception/research/projects/whither-music/>`_



This work has received support from the European Research Council (ERC) under
the European Union’s Horizon 2020 research and innovation programme under grant
agreement No. 670035 project `"Con Espressione" <https://www.jku.at/en/institute-of-computational-perception/research/projects/con-espressione/>`_
and the Austrian Science Fund (FWF) under grant P 29840-G26 (project
`Computer-assisted Analysis of Herbert von Karajan's Musical Conducting Style <https://karajan-research.org/programs/musical-interpretation-karajan>`_ )

.. image:: ./images/aknowledge_logo.png
   :alt: ERC-FWF Logo.
   :align: center

