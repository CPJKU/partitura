Miscelaneous considerations
===========================

- leave match file support for v2.0 (to be released before the tutorial)?

- `load_musicxml` and `load_midi` return lists of `Part` or `PartGroup` elements. To reliably get the first part requires:
    `part = next(iter_parts(load_musicxml('score.xml')), None)`

    This is quite verbose for the simple cases where there is only one
    part. Maybe we should return the part object in case there is only one (not
    embedded in a list), and add an option `return_list` to facilitate handling
    the result in a generic way.

- make `alter` argument of `Note` constructor optional (default `None`)?

- voice problem
  
  Synchronicity within a voice (or when no voice attribute is present) by means of backup/forward is not well supported in musicxml viewers. use of chord tag only works when all simultaneous notes have equal duration.
  Preferred solution: within a set of simultaneous notes make notes of equal duration belong to the same voice (and use chord tag). I.e. in a triad with durations n1:q n2:q n3:h, n1 and n2 would belong to one voice with note n1 including a chord tag, and n3 would be the next voice. Possible post-processing fix to Chew's algorithm: change voice of notes that have same-length vertical neighbor and not adjoint horizontal neighbor to be the same of same-length vertical neighbors.

TODO
----

  - documentation
    - match.py: convert existing docstrings to numpy style

  - load MIDI to score.Part
    - [v] compute symbolic_durations from MIDI durations
    - [v] pitch spelling (implement PS13?)
    - [V] convert mido keysig names (A#m) to fifths+mode representation
    - [V] voice assignment
    - [V] recognize tuplets
    - [V] estimate clef
    - estimate staffs
    - adaptive quantization
    
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
    - [V] fill gaps in measure with rests
    - add measures if they are missing (implemented but needs to be isolated from `import_midi.create_part`)
    - avoid polyphony inside voices
    
 - test
     - [V] musicxml -> score -> musicxml
     - [V] unfold timeline (need more examples?)
     - [V] test case for voice assignment
     - test midi import with time sig change
     
