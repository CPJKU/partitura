"""
This module contains the test cases for testing the note_array attribute of
the Part class.

"""

import unittest

import partitura.score as score
from partitura import load_musicxml
from partitura.utils.music import note_array_from_part
import numpy as np

from . import NOTE_ARRAY_TESTFILES


class TestNoteArray(unittest.TestCase):
    """
    Test the note_array attribute of the Part class
    """

    def test_notearray_1(self):
        part = score.Part("P0", "My Part")

        part.set_quarter_duration(0, 10)
        part.add(score.TimeSignature(3, 4), start=0)
        part.add(score.Note(id="n0", step="A", octave=4), start=0, end=10)

        note_array = part.note_array
        self.assertTrue(len(note_array) == 1)

    def test_notearray_beats(self):
        part = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        note_array = note_array_from_part(part)
        expected_onset_beats = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29, 30, 32]

        self.assertTrue(np.array_equal(note_array["onset_beat"], expected_onset_beats))

    def test_notearray_musical_beats(self):
        part = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        note_array = note_array_from_part(part, use_musical_beat=True)
        expected_onset_beats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14]

        self.assertTrue(np.array_equal(note_array["onset_beat"], expected_onset_beats))

    def test_notearray_ts_beats(self):
        part = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        note_array = note_array_from_part(part, include_time_signature=True)
        expected_musical_beats = [2, 2, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2]
        expected_beats = [6, 6, 9, 9, 9, 12, 12, 12, 12, 2, 2, 2, 2]

        self.assertTrue(
            np.array_equal(note_array["ts_musical_beats"], expected_musical_beats)
        )
        self.assertTrue(np.array_equal(note_array["ts_beats"], expected_beats))


if __name__ == "__main__":
    unittest.main()
