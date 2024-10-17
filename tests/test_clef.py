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
from partitura.score import merge_parts

class TestingClefFeatureExtraction(unittest.TestCase):
    def test_clef_feature_exctraction(self):
        for fn in CLEF_TESTFILES:
            score = load_musicxml(fn, force_note_ids = "keep")
            sna1 = compute_note_array(score.parts[0],
                               feature_functions=["clef_feature"])
            sna2 = compute_note_array(score.parts[1],
                               feature_functions=["clef_feature"])
            mpart = merge_parts(score.parts, reassign="staff")
            sna3 = compute_note_array(mpart,
                               feature_functions=["clef_feature"])
            
            sna1test1 = sna1["clef_feature.clef_sign"] == np.array([1., 0., 2., 0., 0., 2., 0., 0., 1., 0.])
            sna1test2 = sna1["clef_feature.clef_line"] == np.array([4., 2., 3., 2., 2., 4., 2., 2., 4., 2.])
            sna1test3 = sna1["clef_feature.clef_octave_change"] == np.array([0.,  0.,  0.,  1.,  1.,  0., -1.,  0.,  0.,  0.])
            self.assertTrue(np.all(sna1test1), "clef sign does not match")
            self.assertTrue(np.all(sna1test2), "clef line does not match")
            self.assertTrue(np.all(sna1test3), "clef octave does not match")


            sna2test1 = sna2["clef_feature.clef_sign"] == np.array([1., 0.])
            sna2test2 = sna2["clef_feature.clef_line"] == np.array([4., 2.])
            sna2test3 = sna2["clef_feature.clef_octave_change"] == np.array([0.,  0.])
            self.assertTrue(np.all(sna2test1), "clef sign does not match")
            self.assertTrue(np.all(sna2test2), "clef line does not match")
            self.assertTrue(np.all(sna2test3), "clef octave does not match")
            

            
            print(sna3["clef_feature.clef_sign"], sna3["clef_feature.clef_line"], sna3["clef_feature.clef_octave_change"])

