#!/usr/bin/env python
import numpy as np
import logging
import unittest

from partitura import load_musicxml
from partitura.score import merge_parts
from partitura.utils.music import compute_pianoroll, pianoroll_to_notearray

from tests import MERGE_PARTS_TESTFILES

LOGGER = logging.getLogger(__name__)


class TestMergeParts(unittest.TestCase):
    """
    Test merge parts utility

    Asap failing in 
    [('data/raw/asap-dataset/Chopin/Berceuse_op_57/xml_score.musicxml',
  partitura.score.PartGroup),
 ('data/raw/asap-dataset/Liszt/Hungarian_Rhapsodies/6/xml_score.musicxml',
  partitura.score.PartGroup),
 ('data/raw/asap-dataset/Ravel/Gaspard_de_la_Nuit/1_Ondine/xml_score.musicxml',
  list),
 ('data/raw/asap-dataset/Ravel/Miroirs/3_Une_Barque/xml_score.musicxml',
  partitura.score.PartGroup),
 ('data/raw/asap-dataset/Scriabin/Sonatas/5/xml_score.musicxml',
  partitura.score.PartGroup)]
    """

    def test_different_divs(self):
        parts = load_musicxml(MERGE_PARTS_TESTFILES[2])
        self.assertRaises(Exception, merge_parts)

    def test_list_of_parts_and_partgroup(self):
        parts = load_musicxml(MERGE_PARTS_TESTFILES[1])
        merged_part = merge_parts(parts)
        # TODO complete the test
        self.assertTrue(True)
