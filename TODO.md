Things to do before merging develop into master for v1.0.0
----------------------------------------------------------

- Read about `semantic versioning <https://www.jvandemo.com/a-simple-guide-to-semantic-versioning/>`_
- Add MIDI export
- Add option to handle simultaneous/same duration notes in `estimate_voices`
- Remove match.py (and references to it)
- Ensure_list keyword in `load_musicxml`
- Remove this TODO.md
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

