"""
This file contains test functions for music21 import
"""

import unittest


from tests import M21_TESTFILES
from partitura import load_music21, load_mei, EXAMPLE_MEI
import partitura.score as score
from partitura.io.importmei import MeiParser
from partitura.utils import compute_pianoroll
from lxml import etree
from xmlschema.names import XML_NAMESPACE
import music21 as m21

import numpy as np
from pathlib import Path


class TestImportMEI(unittest.TestCase):
    
    def test_grace_note(self):
        m21_score = m21.converter.parse(M21_TESTFILES[1])
        part_list = load_music21(m21_score)
        part = list(score.iter_parts(part_list))[0]
        grace_notes = list(part.iter_all(score.GraceNote))
        self.assertTrue(len(part.note_array()) == 7)
        self.assertTrue(len(grace_notes) == 4)

    def test_clef(self):
        m21_score = m21.converter.parse(M21_TESTFILES[0])
        part_list = load_music21(m21_score)
        # test on part 2
        part2 = list(score.iter_parts(part_list))[2]
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
        part3 = list(score.iter_parts(part_list))[3]
        clefs3 = list(part3.iter_all(score.Clef))
        self.assertTrue(len(clefs3) == 2)
        self.assertTrue(clefs3[0].start.t == 0)
        self.assertTrue(clefs3[1].start.t == 4)
        self.assertTrue(clefs3[1].sign == "G")
        self.assertTrue(clefs3[1].line == 2)
        self.assertTrue(clefs3[1].staff == 4)
        self.assertTrue(clefs3[1].octave_change == -1)