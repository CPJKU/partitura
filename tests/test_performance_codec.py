"""

This file contains test functions for Performance Array Calculations

"""
import logging
import unittest
from tempfile import TemporaryFile

from tests import MATCH_IMPORT_EXPORT_TESTFILES
from partitura import load_match
from partitura.musicanalysis.performance_codec import PerformanceCodec

class TestPerformanceCoded(unittest.TestCase):
    def test_encode(self):
        pc = PerformanceCodec()
        for fn in MATCH_IMPORT_EXPORT_TESTFILES:
            ppart, alignment, spart = load_match(fn, create_part=True)
            performed_array, _ = pc.encode(spart, ppart, alignment)
            print(performed_array[:10])