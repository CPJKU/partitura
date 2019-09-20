Things to do before merging develop into master for v1.0.0
----------------------------------------------------------

- (all) Read about `semantic versioning <https://www.jvandemo.com/a-simple-guide-to-semantic-versioning/>`_

- (tg) Add MIDI export

- [done] (cc) Add option to handle simultaneous/same duration notes in `estimate_voices`

- [done] (cc) Remove match.py (and references to it)

- (mg) Remove instrument_assignment.py (and references to it)

- (?) update/expand README.rst; use madmom README.rst as a guideline (it can be less elaborate) 

- (mg) Ensure_list keyword in `load_musicxml`

- (mg) Remove this TODO.md (save the future todo items somewhere else)

- (all) Check that there are no things in the public API (i.e. in the docs) that are
  not strictly necessary for the intended functionality (use __all__ to select
  which module contents are exposed) -

- (all) Check that all items in the public API (i.e. that appear in the HTML docs) are properly documented.

- (cc) Add reference to Dave's paper at top of pitch_spelling module. Perhaps also reference "Meredith, D. (2019). RecurSIA-RRT: Recursive translatable point-set pattern discovery with removal of redundant translators. 12th International Workshop on Machine Learning and Music. Würzburg, Germany."

- [done] (all) remove all print statements (only use logging)

- [done] (all) get rid of f-strings to relax python >=3.6 version requirement

- (mg, cc) test key estimation

- *add your stuff here*

Todo beyond v1.0.0
------------------

- add match support
- add instrument support
- make `alter` argument of `Note` constructor optional (default `None`)?
- importmidi: estimate staffs
- importmidi: adaptive quantization
- score: support da capo/fine in timeline unfolding
- exportmusicxml: support da capo/fine
- exportmusicxml: warn that musical content outside measures will not be exported (if any)
- exportmusicxml: avoid polyphony inside voices (is this only a musescore problem? should it be optional?)

