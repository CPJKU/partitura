#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Performance Array Calculations
"""
import unittest
import numpy as np
from partitura import load_match
from tests import MATCH_EXPRESSIVE_FEATURES_TESTFILES
from partitura.musicanalysis.performance_features import compute_performance_features
import os



class TestPerformanceFeatures(unittest.TestCase):
    def test_performance_features(self):

        True_array = np.array([('n1', 0.23374297, 0.        ,  0.        , 0.       , 0.        ,  89.74999 , 62.000057, 0., 0.16015087, -0.5, 0.5 , 59, 4.9925 , 0.8775 , 44, 1, 0., 0., 1.4700003),
       ('n4', 0.03011051, 0.07375002, -0.20350015, 0.3837298, 0.03447518, 114.25004 , 61.000244, 0., 0.4027142 ,  0. , 1.  , 40, 5.7025 , 2.4375 , 22, 7, 0., 0., 2.8474998),
       ('n3', 2.527984  , 0.07375002, -0.20350015, 0.3837298, 0.03447518,  87.500046, 61.000244, 0., 0.4027142 ,  0. , 0.25, 56, 5.77625, 2.36375, 26, 3, 0., 0., 2.8474998)],
        dtype=[('id', '<U256'), 
             ('articulation_feature.kor', '<f4'),
             ('asynchrony_feature.delta', '<f4'), 
             ('asynchrony_feature.pitch_cor', '<f4'), 
             ('asynchrony_feature.vel_cor', '<f4'), 
             ('asynchrony_feature.voice_std', '<f4'), 
             ('pedal_feature.onset_value', '<f4'), 
             ('pedal_feature.offset_value', '<f4'), 
             ('pedal_feature.to_prev_release', '<f4'), 
             ('pedal_feature.to_next_release', '<f4'), 
             ('onset', '<f4'), 
             ('duration', '<f4'), 
             ('pitch', '<i4'), 
             ('p_onset', '<f4'), 
             ('p_duration', '<f4'),
             ('velocity', '<i4'), 
             ('voice', '<i4'), 
             ('slur_feature.slur_incr', '<f4'), 
             ('slur_feature.slur_decr', '<f4'), 
             ('beat_period', '<f4')])
        fn = MATCH_EXPRESSIVE_FEATURES_TESTFILES[0]
        perf, alignment, score = load_match(filename=fn, create_score=True)
        features = compute_performance_features(score, 
                                                perf,
                                                alignment,
                                                feature_functions = "all")
        
        self.assertTrue(np.all(True_array == features[:3]), f"The expression features don't match the original.")
