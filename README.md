# Partitura

Paritura is a toolkit for handling

* Match files
* MusicXML files
* Worm Files
* MIDI files (coming soon)

**Disclaimer**: Not all types of files are equally thoroughly supported. Furthermore, there is a utils module which contains miscelaneous utilities. 

MusicXML
========

The following elements that are extracted from the MusicXML:

* Page
* System
* Measure
* Note
  * Slur
  * Tie (implicit: create one note for all tied notes)
* Time Signature
* Key Signature
* Transposition
* Tempo (in `<sound>`)
* predefined dynamics directions (p, pp, f, wedges, ...)
* tempo and dynamics directions notated as `<words>` (rallentando, crescendo, ...)
* Repeats (all types?)

unsupported (for now):

* unpitched notes
* visual attributes like placement, x-offset
