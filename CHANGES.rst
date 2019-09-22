Release Notes
=============

Version 0.1.2 (prerelease; unreleased)
--------------------------------------

API changes:

* Treat missing key signature mode as major
* Function `iter_parts` accepts non-list arg
* Don't do quantization by default

Bug fixes:

* Avoid sharing symbolic_duration dictionaries between notes
* Rework MIDI loading: do not accumulate quantization errors
* Make sure last tied note actually gets tied
* Do not populate symbolic_duration with None when values are missing

Other changes:
  
* Add/improve documentation
* Show triplets in pretty()
* Remove trailing whitespace

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
