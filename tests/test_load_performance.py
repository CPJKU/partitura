"""

This file contains test functions for the load_performance method

"""
import unittest

from tests import MATCH_IMPORT_EXPORT_TESTFILES


from partitura import load_performance, EXAMPLE_MIDI
from partitura.io import NotSupportedFormatError
from partitura.performance import PerformedPart, Performance


class TestLoadScore(unittest.TestCase):
    def test_load_performance(self):
        for fn in MATCH_IMPORT_EXPORT_TESTFILES + [EXAMPLE_MIDI]:
            load_performance(fn)

    def load_performance(self, fn):
        try:
            performance = load_performance(fn)
            self.assertTrue(isinstance(performance, Performance))
        except NotSupportedFormatError:
            self.assertTrue(False)
