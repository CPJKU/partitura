Introduction
============

The principal aim of the `partitura` package is to handle richly structured
musical information as conveyed by common Western music notation. It provides a
much wider range of possibilities to deal with music than the more reductive
(but very common) pianoroll-oriented approach inspired by the MIDI standard.

Specifically, the package allows for representing a variety of information in
musical scores beyond the onset, duration and MIDI pitch numbers of notes, such
as:

* pitch spellings,
* symbolic duration categories,
* and voicing information.

Moreover, it supports musical notions that are not note-related, like:

* measures,
* tempo indications,
* performance directions,
* repeat structures,
* and time/key signatures.

Supported file types
--------------------

Musical data can be loaded from and saved to MusicXML files, as well as MIDI
files. 

Relation to `music21 <https://web.mit.edu/music21/>`_
-----------------------------------------------------

The aims of `partitura` are much more modest than those of `music21`, which aims
to provide toolkit for computer-aided musicology. Instead, `partitura` provides
functionality for python programmers to work with musical data like MusicXML
files. Although it does contain some tools for music analysis, such as pitch
spelling and voice assignment, these tools are intended to fill in missing
information with plausible values, for instance when loading a score from a MIDI
file.
