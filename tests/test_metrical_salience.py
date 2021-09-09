"""

This file contains test functions for the metrical salience computation

"""

import unittest
from partitura import load_musicxml
from partitura.utils.music import note_array_from_part

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
        metrical_positions = part.metrical_position_map([n.start.t for n in notes])
        expected_rel_positions = [
            0,
            8,
            0,
            0,
            24,
            28,
            32,
            40,
            0,
            0,
            24,
            48,
            56,
            0,
            16,
            32,
            33,
            34,
            36,
            40,
            48,
            0,
            16,
        ]
        expected_measure_duration = (
            [16 for i in range(2)]
            + [48 for i in range(6)]
            + [64 for i in range(5)]
            + [72 for i in range(8)]
            + [32 for i in range(2)]
        )
        # test relative positions
        self.assertTrue(
            np.array_equal(metrical_positions[:, 0], expected_rel_positions)
        )  # test measure durations
        self.assertTrue(
            np.array_equal(metrical_positions[:, 1], expected_measure_duration)
        )

    def test_metrical_salience_map(self):
        part = load_musicxml(METRICAL_SALIENCE_TESTFILES[0])
        note_array = note_array_from_part(part, include_metrical_salience=True)
        expected_notes_on_downbeat = [0, 2, 3, 8, 9, 13, 21]
        expected_rel_positions = [
            0,
            8,
            0,
            0,
            24,
            28,
            32,
            40,
            0,
            0,
            24,
            48,
            56,
            0,
            16,
            32,
            33,
            34,
            36,
            40,
            48,
            0,
            16,
        ]
        expected_measure_duration = (
            [16 for i in range(2)]
            + [48 for i in range(6)]
            + [64 for i in range(5)]
            + [72 for i in range(8)]
            + [32 for i in range(2)]
        )
        # test if all downbeats are in the right place
        self.assertTrue(
            np.array_equal(
                np.flatnonzero(note_array["is_downbeat"]), expected_notes_on_downbeat
            )
        )
        self.assertTrue(
            np.array_equal(note_array["rel_onset_div"], expected_rel_positions)
        )
        self.assertTrue(
            np.array_equal(note_array["tot_measure_div"], expected_measure_duration,)
        )


if __name__ == "__main__":
    unittest.main()
