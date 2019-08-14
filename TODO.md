TODO
====

  - load MIDI to score.Part
    - compute symbolic_durations from MIDI durations (partly done, see score.estimate_symbolic_duration)
    - pitch spelling (implement PS13?)
    - convert mido keysig names (A#m) to fifths+mode representation
    - [V] voice assignment
    - test case for voice assignment
    
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
    - tempo
    - new page/system
    
 - test
     - [V] musicxml -> score -> musicxml
     - [V] unfold timeline (need more examples?)
     
