Release Notes
=============

Version 0.4.0 (Released on 2021-05-28)
--------------------------------------

API changes:

* Different format for `note_array` attribute (in `Part`, `PerformedPart` and `PartGroup`). The name of the fields in the note arrays for onset and duration information now include the units: `beat`, `quarter`, and `div` for scores and `sec` for performances.
* The music analysis functions now accept `Part` and `PerformedPart` as well as note arrays
* Support for all MIDI controls in `PerformedPart`. The controls are only stored by their CC number (not name).
* Add default program change option to `PerformedPart`

  
New features:

* Add function to create 2D pianoroll representations from `Part`, `PerformedPart` or note arrays.
* Add document order index to parts read from MusicXML
* Support for sustain pedal info in MusicXML files
* Add tonal tension profiles by Herremans and Chew (2016)
* Add key_signature_map to Part objects
* Add read support for Match and Corresp files of Nakamura's music alignment tool
* Add `sanitize_part` function removing incomplete slurs and tuplets.

Bug fixes:
  
* Fix tempo change bug in load_performance_midi
* Fix saving parts with anacrusis to score MIDI
* When defining `Note` objects, lower-case note names are converted to upper-case. Before this the behavior was undefined when using lower-case note names
* Fix bug in pattern for matchfile
* Fix issue with MIDI velocity for overlapping notes in piano rolls
* Fix bug in `add_measures`
* Fix GraceNote bug

Other changes:

* Add documentation
* Code cleanup

Version 0.3.5 (Released on 2019-11-08)
--------------------------------------

Other changes:

* Add documentation


Version 0.3.4 (Released on 2019-11-08)
--------------------------------------

API changes:

* Rename `out_fmt` kwarg to `fmt` in `show`
* Add `dpi` kwarg to `show`
* Rename `show` function to `render`

New features:

* Save rendered scores to image file using `render`
* Add `wedge` attribute to DynamicLoudnessMarkings to differentiate them
  from textual directions

Bug fixes:
  
* Do not crash when calling time_signature_map on empty Part instances
* Fix bug in repeat unfolding


Version 0.3.3 (Released on 2019-11-04)
--------------------------------------

Bug fixes:
  
* Fix missing dedent import


Version 0.3.2 (Released on 2019-11-03)
--------------------------------------

API changes:

* More systematic direction ontology

New features:

* Add `grace_seq_len` property to grace note
* Add `backwards` keyword arg to `iter_grace_seq`

Bug fixes:
  
* Fix regression in slur handling; Remove unended objects in import


Version 0.3.1 (Released on 2019-10-30)
--------------------------------------

API changes:

* Rename load_midi -> load_score_midi
* Rename xml_to_notearray -> musicxml_to_notearray
* Rename divisions_map -> quarter_duration_map
* Page/System: rename nr -> number
* Make Part.points private (_points)
* Remove voice related methods on Note (only used in obsolete expand_grace_notes
version)

New features:

* Save Part as MIDI file (save_score_midi)
* Add PerformedPart to represent performances
* Load MIDI as PerformedPart (load_performance_midi)
* Save PerformedPart as MIDI (save_performance_midi)
* Add support for Match files (load_match)
* New options for iter_current_next
* Make_measures respects existing measures optionally
* Export articulations to musicxml.
* load_score_midi: Estimate staffs (if not specified)
* load_score_midi: Estimate voices (if not specified)
* load_musicxml: Set end times of score elements to start of next element of
  that class (Page, System, Direction)
* Define Incr/Decr loudness/tempo directions
* Add iter_prev/iter_next methods
* Be explicit about kwargs in Note creation
* Add show function to display score, using either MuseScore or Lilypond as
  backend
* Add load_via_musescore to load scores in other formats 

Bug fixes:

* Better clef support in musicxml export
* export_musicxml: fixes in handle wedge/dashes export
* The order in which simulatenous notes are listed in a timepoint no longer
  influences the chord-handling logic in voice estimation, and the musicxml
  export.
* Fix incorrect construction of dtypes for structarray in voice_separation
* Fix in anacrusis handling
* Fix in iter_current_next
* import_musicxml: check for <backup> crossing measure boundary
    
Other changes:

* Get rid of deprecated get_prev/next_of_type
* Tuplet/Slur: make use of getter/setter for start/end_note
* Improvements in parse_direction
* expand_grace_notes now simpy sets note durations, without shifting onsets
* Rename strictly_monophonic_voices keyword arg to monophonic_voices in
  estimate_voices, and implement (previously unimplemented) functionality: With
  monophonic_voice=False, notes with same onset and duration as treated as
  chords and assigned to the same voice
* More documentation

Version 0.2.0 (prerelease; Released on 2019-10-04)
--------------------------------------------------

API changes:

* The TimeLine class has been merged into the Part class
  
New features:

* Add `find_tuplets` and `tie_notes` to public API
* New Tuplet class analog to Slur, allows for better musicxml tuplet
  support
* Remove deprecated get_starting_objects_of_type/get_ending_objects_of_type (use
  iter_starting/iter_ending)

Bug fixes:

* Multiple fixes in tuplet and slur handling 

Other changes:

* Update package description/long description
* More documentation
* Add separate tuplet and slur test cases
* Improve show_diff


Version 0.1.2 (prerelease; Released on 2019-09-29)
--------------------------------------------------

API changes:

* New approach to handling divisions
* Treat missing key signature mode as major
* Function `iter_parts` accepts non-list arg
* Don't do quantization by default
* Change make alter a keyword arg in Note constructor
* Remove `parse_words` from API
* Export part-groups to musicxml
* Add PartGroup constructor keyword args
* Rename PartGroup.name -> PartGroup.group_name (for consistency)
* Rename Part.part_id -> Part.id
* `iter_parts` accepts non-list arg
* Remove `Measure.upbeat` property (use `Measure.incomplete`)

New features:

* New add_measures function to automatically add measures to a Part
* Add inverted quarter/beat map

Bug fixes:

* Avoid sharing symbolic_duration dictionaries between notes
* Rework MIDI loading: do not accumulate quantization errors
* Make sure last tied note actually gets tied
* Do not populate symbolic_duration with None when values are missing
* When exporting to musicxml, avoid polyphony within voices by reassigning notes to new voices where necessary
* Filter null characters when exporting musicxml to avoid lxml exception
* Loggin: info -> debug
* Don't use divisions_map
* Fix leftover references to old API
* Fix `add_measures`
* Handle part/group names when importing MIDI
* Fix bug in `divisions_map`
* fix bug in `estimate_symbolic_duration`
  
Other changes:
  
* Add test case for beat maps and symbolic durations
* Improve direction parsing
* Remove polyphony within voices when exporting to musicxml
* Add show function to show typeset score (using lilypondn)
* Add/improve documentation
* Improve pretty printing
* Remove trailing whitespace
* More exhaustive tuplet search
* Write tests for tuplet detection
* Write tests for importmidi assignment modes
* Rewrite quarter/beat map construction
* Create (non-public API) utils sub package

Version 0.1.1 (prerelease)
--------------------------
Bug fixes:

* Tweak docs/conf.py to work correctly on readthedocs.org

Other changes:
  
* Fix incorrect version in setup.py

Version 0.1.0 (prerelease)
--------------------------

This is the first prerelease of the package. In this release MIDI export
functionality is missing and the documentation is incomplete.
