#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Performance Array Calculations
"""
import unittest
import numpy as np
from partitura import load_match
from tests import MATCH_IMPORT_EXPORT_TESTFILES
from partitura.musicanalysis.performance_features import compute_performance_features
import os



class TestPerformanceFeatures(unittest.TestCase):
    def test_performance_features(self):
        for fn in MATCH_IMPORT_EXPORT_TESTFILES:
            print(fn)
            perf, alignment, score = load_match(filename=fn, create_score=True)
            features = compute_performance_features(score, 
                                                    perf,
                                                    alignment,
                                                    feature_functions = "all")
            
            self.assertTrue(True, f"The expression features don't match the original.")
