Release Notes
=============

Version 0.2.1 (prerelease)
--------------------------


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
