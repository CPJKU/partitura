#!/usr/bin/env python
# -*- coding: utf-8 -*-

#%%
"""
This module contains tests for measures with non-integer IDs in musicxml. 

Such measure IDs can occur in irregular measures (i.e., pickups in the middle of the piece, i.e. variations pieces, measures with fermata or cadenza ornamentations.

Fix: added 'name' property to measures to reflect non-integer measures.
"""
import unittest

import numpy as np
from partitura.io import load_musicxml
from partitura.score import *

from tests import MUSICXML_PARTIAL_MEASURES_TESTFILES


class TestPartialMeasures(unittest.TestCase):
    """
    Test parsing of musicxml files with single partial/irregular measures
    """
    
    def test_measure_number_name_single(self):
        sc = load_musicxml(MUSICXML_PARTIAL_MEASURES_TESTFILES[0])
        spart = sc.parts[0]
        spart_variants = make_score_variants(spart)
        spart_unfolded = spart_variants[0].create_variant_part()
        
        unfolded_measure_numbers = [m.number for m in spart_unfolded.measures]
        unfolded_measure_names = [m.name for m in spart_unfolded.measures]
        
        expected_unfolded_measure_numbers = [1,2,3,1,2,4,5,6,7]
        expected_unfolded_measure_names = ['1','2','3','1','2','X1','4','X2','5']
        
        self.assertTrue(
            np.array_equal(unfolded_measure_numbers, expected_unfolded_measure_numbers)
        )
        self.assertTrue(
            np.array_equal(unfolded_measure_names, expected_unfolded_measure_names)
        )

    def test_measure_number_name_consecutive(self):
        sc = load_musicxml(MUSICXML_PARTIAL_MEASURES_TESTFILES[1])
        spart = sc.parts[0]
        
        measure_numbers = [m.number for m in spart.measures]
        measure_names = [m.name for m in spart.measures]
        
        expected_measure_numbers = [1,2,3,4,5,6,7]
        expected_measure_names = ['1','2','3','X1','X2','X3','4']
        
        self.assertTrue(
            np.array_equal(measure_numbers, expected_measure_numbers)
        )
        self.assertTrue(
            np.array_equal(measure_names, expected_measure_names)
        )

if __name__ == "__main__":
    unittest.main()