"""
This module contains the test cases for testing the note_array attribute of
the Part class.

"""

import unittest

import partitura.score as score
from partitura import load_musicxml, load_kern
from partitura.utils.music import note_array_from_part, ensure_notearray
import numpy as np

from tests import NOTE_ARRAY_TESTFILES, KERN_TESFILES


class TestNoteArray(unittest.TestCase):
    """
    Test the note_array attribute of the Part class
    """

    def test_notearray_1(self):
        part = score.Part("P0", "My Part")

        part.set_quarter_duration(0, 10)
        part.add(score.TimeSignature(3, 4), start=0)
        part.add(score.Note(id="n0", step="A", octave=4), start=0, end=10)

        note_array = part.note_array()
        self.assertTrue(len(note_array) == 1)

    def test_notearray_beats(self):
        score = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        note_array = score.note_array()
        expected_onset_beats = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29, 30, 32]

        self.assertTrue(np.array_equal(note_array["onset_beat"], expected_onset_beats))

    def test_notearray_musical_beats1(self):
        part = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        part.use_musical_beat()
        note_array = note_array_from_part(part)
        expected_onset_beats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14]

        self.assertTrue(np.array_equal(note_array["onset_beat"], expected_onset_beats))
        self.assertTrue(
            np.array_equal(part.note_array()["onset_beat"], expected_onset_beats)
        )

    def test_use_musical_beats1(self):
        part = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        part.use_musical_beat({"6/8": 3, "2/4": 1})
        note_array = note_array_from_part(part)
        expected_onset_beats = [0, 1.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11.5, 12.5]

        self.assertTrue(np.array_equal(note_array["onset_beat"], expected_onset_beats))
        self.assertTrue(np.array_equal(part.note_array(), note_array_from_part(part)))

    def test_use_musical_beats2(self):
        part = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        # set musical beats
        part.use_musical_beat()
        note_array = part.note_array()
        self.assertTrue(part._use_musical_beat == True)
        # unset musical beats
        part.use_notated_beat()
        self.assertTrue(part._use_musical_beat == False)
        self.assertFalse(np.array_equal(part.note_array, note_array))

    def test_notearray_ts_beats(self):
        part = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        note_array = note_array_from_part(part, include_time_signature=True)
        expected_beats = [6, 6, 9, 9, 9, 12, 12, 12, 12, 2, 2, 2, 2]
        self.assertTrue(np.array_equal(note_array["ts_beats"], expected_beats))

        # now using musical beats
        part.use_musical_beat()
        note_array = note_array_from_part(part, include_time_signature=True)
        expected_musical_beats = [2, 2, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2]
        self.assertTrue(np.array_equal(note_array["ts_mus_beats"], expected_musical_beats))

    def test_ensure_na_different_divs(self):
        # check if divs are correctly rescaled when producing a note array from 
        # parts with different divs values
        parts = list(score.iter_parts(load_kern(KERN_TESFILES[7])))
        # note_arrays = [p.note_array(include_divs_per_quarter= True) for p in parts]
        merged_note_array = ensure_notearray(parts)
        for note in merged_note_array[-4:]:
            self.assertTrue(note["onset_div"] == 92)
            self.assertTrue(note["duration_div"] == 4)
            self.assertTrue(note["divs_pq"] == 4)




if __name__ == "__main__":
    unittest.main()
