"""
This file contains test functions for music21 import
"""
import unittest

from tests import M21_TESTFILES
from partitura import load_music21
import partitura.score as score
import partitura as pt

import music21 as m21
import numpy as np


class TestImportM21(unittest.TestCase):
    def test_grace_note(self):
        m21_score = m21.converter.parse(M21_TESTFILES[1])
        pt_score = load_music21(m21_score)
        part = pt_score.parts[0]
        grace_notes = list(part.iter_all(score.GraceNote))
        self.assertTrue(len(part.note_array()) == 7)
        self.assertTrue(len(grace_notes) == 4)

    def test_clef(self):
        m21_score = m21.converter.parse(M21_TESTFILES[0])
        pt_score = load_music21(m21_score)
        # test on part 2
        part2 = pt_score.parts[2]
        clefs2 = list(part2.iter_all(score.Clef))
        self.assertTrue(len(clefs2) == 2)
        self.assertTrue(clefs2[0].start.t == 0)
        self.assertTrue(clefs2[0].sign == "C")
        self.assertTrue(clefs2[0].line == 3)
        self.assertTrue(clefs2[0].staff == 3)
        self.assertTrue(clefs2[0].octave_change == 0)
        self.assertTrue(clefs2[1].start.t == 8)
        self.assertTrue(clefs2[1].sign == "F")
        self.assertTrue(clefs2[1].line == 4)
        self.assertTrue(clefs2[1].staff == 3)
        self.assertTrue(clefs2[1].octave_change == 0)
        # test on part 3
        part3 = pt_score.parts[3]
        clefs3 = list(part3.iter_all(score.Clef))
        self.assertTrue(len(clefs3) == 2)
        self.assertTrue(clefs3[0].start.t == 0)
        self.assertTrue(clefs3[1].start.t == 4)
        self.assertTrue(clefs3[1].sign == "G")
        self.assertTrue(clefs3[1].line == 2)
        self.assertTrue(clefs3[1].staff == 4)
        self.assertTrue(clefs3[1].octave_change == -1)

    # check if the note array computed directly from partitura is the same as the one computed by importing first with m21
    def test_note_array1(self):
        # load score from music21
        m21_score = m21.converter.parse(M21_TESTFILES[0])
        pt_score_from_m21 = load_music21(m21_score)
        # load score directly from partitura
        pt_score_direct = pt.load_score(M21_TESTFILES[0])

        # compare the note arrays
        note_array_from_m21 = pt_score_from_m21.note_array()
        note_array_direct = pt_score_direct.note_array()
        self.assertTrue(
            np.array_equal(note_array_from_m21["pitch"], note_array_direct["pitch"])
        )
        self.assertTrue(
            np.array_equal(
                note_array_from_m21["onset_beat"], note_array_direct["onset_beat"]
            )
        )
        self.assertTrue(
            np.array_equal(
                note_array_from_m21["duration_beat"], note_array_direct["duration_beat"]
            )
        )

    def test_note_array2(self):
        # load score from music21
        m21_score = m21.converter.parse(M21_TESTFILES[2])
        pt_score_from_m21 = load_music21(m21_score)
        # load score directly from partitura
        pt_score_direct = pt.load_score(M21_TESTFILES[2])

        # compare the note arrays
        note_array_from_m21 = pt_score_from_m21.note_array()
        note_array_direct = pt_score_direct.note_array()
        self.assertTrue(
            np.array_equal(note_array_from_m21["pitch"], note_array_direct["pitch"])
        )
        self.assertTrue(
            np.array_equal(
                note_array_from_m21["onset_beat"], note_array_direct["onset_beat"]
            )
        )
        self.assertTrue(
            np.array_equal(
                note_array_from_m21["duration_beat"], note_array_direct["duration_beat"]
            )
        )
        self.assertTrue(
            np.array_equal(
                note_array_from_m21["onset_quarter"], note_array_direct["onset_quarter"]
            )
        )
        self.assertTrue(
            np.array_equal(
                note_array_from_m21["duration_quarter"],
                note_array_direct["duration_quarter"],
            )
        )

    def test_note_array3(self):
        # load score from music21
        m21_score = m21.converter.parse(M21_TESTFILES[3])
        pt_score_from_m21 = load_music21(m21_score)
        # load score directly from partitura
        pt_score_direct = pt.load_score(M21_TESTFILES[3])

        # compare the note arrays
        note_array_from_m21 = pt_score_from_m21.note_array()
        note_array_direct = pt_score_direct.note_array()
        self.assertTrue(
            np.array_equal(note_array_from_m21["pitch"], note_array_direct["pitch"])
        )
        self.assertTrue(
            np.array_equal(
                note_array_from_m21["onset_beat"], note_array_direct["onset_beat"]
            )
        )
        self.assertTrue(
            np.array_equal(
                note_array_from_m21["duration_beat"], note_array_direct["duration_beat"]
            )
        )
