#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for the `load_performance` method
"""
import unittest
import numpy as np
from tests import MOZART_VARIATION_FILES


from partitura import load_performance_midi, EXAMPLE_MIDI
from partitura.io import NotSupportedFormatError
from partitura.performance import Performance


class TestLoadPerformance(unittest.TestCase):
    def test_load_performance(self):
        for fn in [MOZART_VARIATION_FILES["midi"]] + [EXAMPLE_MIDI]:
            try:
                print(fn)
                performance = load_performance_midi(fn)
                self.assertTrue(isinstance(performance, Performance))
            except NotSupportedFormatError:
                self.assertTrue(False)

    def test_array_performance(self):
        for fn in [EXAMPLE_MIDI]:
            performance = load_performance_midi(fn)
            na = performance.note_array()
            self.assertTrue(np.all(na["onset_sec"] * 24 == na["onset_tick"]))


if __name__ == "__main__":
    unittest.main()
