# Partitura

Paritura is a toolkit for handling

* MusicXML files
* MIDI files
* Match files (in v2.0?)

MusicXML
========

The following elements that are extracted from the MusicXML:

* Page
* System
* Measure
* Note
  * Slur
  * Tie
* Time Signature
* Key Signature
* Transposition
* Tempo (in `<sound>`)
* Page and System layout
* canonical dynamics directions (p, pp, f, wedges, ...)
* tempo and dynamics directions notated as `<words>` (rallentando, crescendo, ...)
* tempo specifications in `<sound>` tags
* Repeat signs (including 1/2 endings, but not da capo/fine/segno)

unsupported (for now):

* unpitched notes
* visual attributes like placement, x-offset
* beams
* note stem directions
