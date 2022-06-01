"""

This file contains test functions for Performance Array Calculations

"""
import unittest
import numpy as np
from tests import MATCH_IMPORT_EXPORT_TESTFILES
from partitura import load_match
from partitura.musicanalysis import encode_performance, decode_performance


class TestPerformanceCoded(unittest.TestCase):
    def test_encode_decode(self):
        for fn in MATCH_IMPORT_EXPORT_TESTFILES:
            ppart, alignment, spart = load_match(fn, create_part=True)

            performance_array, _ = encode_performance(spart, ppart, alignment)
            decoded_ppart, decoded_alignment = decode_performance(spart, performance_array, return_alignment=True)
            # normalize ppart notearray onset sec starting from 0.
            orig_sec_array = ppart.note_array()["onset_sec"] - ppart.note_array()["onset_sec"].min()
            target = np.all(np.allclose(np.sort(decoded_ppart.note_array()["onset_sec"]), np.sort(orig_sec_array)))
            self.assertTrue(target, "The decoded Performed Part doesn't match the original.")
