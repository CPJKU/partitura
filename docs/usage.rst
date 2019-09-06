Usage
=====

A musical score can be loaded from a MusicXML file `score.xml` as follows:

>>> from partitura import load_musicxml
>>> scr = load_musicxml('score.xml')

Loading from a MIDI file `score.mid` works in the same way:

>>> from partitura import load_midi
>>> scr = load_mid('score.mid')

The result stored in `score` is a list of `Part` or `PartGroup` objects, depending on the contents of the file.

>>> import partitura.score as score
>>> part = next(iter_parts(scr))

>>> notes = part.list_all(score.Note)

You can also build a score from scratch, by creating a `Part` object:

>>> part = score.Part('My Part')

creating contents explicitly:

>>> divs = score.Divisions(10)
>>> ts = score.TimeSignature(3, 4)
>>> measure1 = score.Measure(number=1)
>>> note1 = score.Note(step='A', alter=None, octave=4)
>>> note2 = score.Note(step='C', alter='#', octave=5)

and adding them to the part:

>>> part.add(0, divs)
>>> part.add(0, ts)
>>> part.add(0, measure1, end=30)
>>> part.add(0, note1, end=15)
>>> part.add(0, note2, end=20)

>>> print(part.pretty())
>>> save_musicxml(part, 'out.xml')
