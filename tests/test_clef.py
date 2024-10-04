#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for clef related methods.
"""
import unittest
from tests import (
    CLEF_TESTFILES
)
from partitura import load_musicxml
from partitura.musicanalysis import compute_note_array
from partitura.musicanalysis.note_features import clef_feature

class TestingClefFeatureExtraction(unittest.TestCase):
    def test_feature_exctraction(self):
        for fn in CLEF_TESTFILES:
            score = load_musicxml(fn)
            sna1 = compute_note_array(score.parts[0],
                               feature_functions=["clef_feature"])
            sna2 = compute_note_array(score.parts[1],
                               feature_functions=["clef_feature"])

