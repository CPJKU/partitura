Release Notes
=============

Version 0.1.2 (prerelease; unreleased)
--------------------------------------

TODO: list changes from 9898e2f8c4b7cb87df650993e57fe3ab28a6a889 (inclusive) upward

API changes:

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
  
* Remove polyphony within voices when exporting to musicxml
* Add show function to show typeset score (using lilypondn)
* Add/improve documentation
* improve pretty printing
* Remove trailing whitespace
* More exhaustive tuplet search
* Write tests for tuplet detection
* Write tests for importmidi assignment modes


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
