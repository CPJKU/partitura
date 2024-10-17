#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for clef related methods.
"""
import unittest
from tests import (
    CLEF_TESTFILES
)
import numpy as np
from partitura import load_musicxml
from partitura.musicanalysis import compute_note_array
from partitura.musicanalysis.note_features import clef_feature

class TestingClefFeatureExtraction(unittest.TestCase):
    def test_clef_feature_exctraction(self):
        for fn in CLEF_TESTFILES:
            score = load_musicxml(fn, force_note_ids = "keep")
            sna1 = compute_note_array(score.parts[0],
                               feature_functions=["clef_feature"])
            sna2 = compute_note_array(score.parts[1],
                               feature_functions=["clef_feature"])
            

            sna1test1 = sna1["clef_feature.clef_sign"] == np.array([1., 0., 2., 0., 0., 2., 0., 0., 1., 0.])
            self.assertTrue(np.all(sna1test1), "clef sign does not match")
            print(sna1["clef_feature.clef_sign"], sna1["clef_feature.clef_line"], sna1["clef_feature.clef_octave_change"])

