#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains the test cases for testing the note_array attribute of
the Part class.
"""

import unittest
import partitura.score as score
from partitura import load_musicxml
import numpy as np
from tests import REST_ARRAY_TESTFILES


class TestNoteArray(unittest.TestCase):
    """
    Test the note_array attribute of the Part class
    """

    def test_restarray_1(self):
        part = score.Part("P0", "My Part")

        part.set_quarter_duration(0, 10)
        part.add(score.TimeSignature(3, 4), start=0)
        part.add(score.Note(id="n0", step="A", octave=4), start=0, end=10)
        part.add(score.Rest(id="r0"), start=10, end=20)

        note_array = part.rest_array()
        self.assertTrue(len(note_array) == 1)

    def test_rest_array(self):
        sc = load_musicxml(REST_ARRAY_TESTFILES[0])
        sc[0].use_musical_beat()
        rest_array = sc[0].rest_array(
            include_metrical_position=True, include_staff=True
        )
        expected_musical_beats = [14, 18]
        self.assertTrue(
            np.array_equal(rest_array["onset_beat"], expected_musical_beats)
        )

    def test_rest_collapse(self):
        sc = load_musicxml(REST_ARRAY_TESTFILES[1])
        rest_array_non_collapsed = sc[0].rest_array()
        expected_length_non_collapsed = 7
        rest_array = sc[0].rest_array(collapse=True)
        expected_length_collapsed = 5
        self.assertTrue(len(rest_array_non_collapsed) == expected_length_non_collapsed)
        self.assertTrue(len(rest_array) == expected_length_collapsed)


if __name__ == "__main__":
    unittest.main()
