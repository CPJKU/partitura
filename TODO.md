TODO
====

Miscelaneous considerations:

- leave match file support for v2.0 (to be released before the tutorial)?

- `load_musicxml` and `load_midi` return lists of `Part` or `PartGroup` elements. To reliably get the first part requires:
    `part = next(iter_parts(load_musicxml('score.xml')), None)`

    This is quite verbose for the simple cases where there is only one
    part. Maybe we should return the part object in case there is only one (not
    embedded in a list), and add an option `return_list` to facilitate handling
    the result in a generic way.

- make `alter` argument of `Note` constructor optional (default `None`)?



TODO
----

  - documentation
    - match.py: convert existing docstrings to numpy style

  - load MIDI to score.Part
    - compute symbolic_durations from MIDI durations (partly done, see score.estimate_symbolic_duration)
    - pitch spelling (implement PS13?)
    - convert mido keysig names (A#m) to fifths+mode representation
    - [V] voice assignment
    
  - load match/score format to score.Part

  - export score.Part to MIDI

  - export score.Part to MusicXML
    - [V] measure
    - [V] divisions
    - [V] key sig
    - [V] time sig
    - [V] directions
	- [V] slurs
	- [V] tied notes
	- [V] grace notes
    - [X] cue notes 
	- [V] accents
    - da capo/fine
    - [V] tempo
    - [V] new page/system
    
 - test
     - [V] musicxml -> score -> musicxml
     - [V] unfold timeline (need more examples?)
     - test case for voice assignment
