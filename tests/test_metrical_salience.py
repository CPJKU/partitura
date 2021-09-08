"""

This file contains test functions for the metrical salience computation

"""

import logging
import unittest
from partitura import load_musicxml
from partitura.utils.music import note_array_from_part
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

    def test_metrical_position_map(self):
        part = load_musicxml(METRICAL_SALIENCE_TESTFILES[0])
        notes = part.notes
        metrical_positions_for_notes = part.metrical_position_map(
            [n.start.t for n in notes]
        )
        # TODO : complete the test
        self.assertTrue(False)

    def test_metrical_salience_map(self):
        part = load_musicxml(METRICAL_SALIENCE_TESTFILES[0])
        note_array = note_array_from_part(part, include_metrical_salience=True)
        # TODO : complete the test
        self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
