"""

This file contains test functions for the metrical salience computation

"""

import logging
import unittest
from partitura import load_musicxml
from pathlib import Path

import numpy as np

from . import METRICAL_SALIENCE_TESTFILES


class TestMetricalSalience(unittest.TestCase):
    def test_measure_map(self):
        part = load_musicxml(METRICAL_SALIENCE_TESTFILES[0])
        notes = part.notes
        measures_for_notes = part.measure_map([n.start.t for n in notes])
        expected_measures_for_notes = np.array(  # the start and end of measures that we expect
            [(0, 16) for i in range(2)]
            + [(16, 64) for i in range(6)]
            + [(64, 128) for i in range(5)]
            + [(128, 200) for i in range(8)]
            + [(200, 232) for i in range(2)]
        )
        self.assertTrue(np.array_equal(measures_for_notes, expected_measures_for_notes))

    def test_saliency_map(self):
        self.assertTrue


if __name__ == "__main__":
    unittest.main()
