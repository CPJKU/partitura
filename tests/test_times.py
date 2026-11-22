"""
This module contains tests for testing conversions from beats and quarters.
"""

import unittest

import numpy as np

import partitura.score as score
from partitura.utils.music import seconds_to_midi_ticks, midi_ticks_to_seconds


def _test_time_pairs(part, time_pairs):
    bm = part.beat_map
    ibm = part.inv_beat_map
    qm = part.quarter_map
    iqm = part.inv_quarter_map
    for tb, tq in time_pairs:
        assert qm(ibm(tb)) == tq
        assert bm(iqm(tq)) == tb


class TestBeatVsQuarterTimes(unittest.TestCase):
    def test_times_1(self):
        # 4/4 anacrusis
        part = score.Part("id")
        # 1 div is 1 quarter
        part.set_quarter_duration(0, 1)
        # 4/4 at t=0
        part.add(score.TimeSignature(4, 4), 0)

        # ANACRUSIS
        # quarter note from t=0 to t=1
        part.add(score.Note("c", 4), 0, 1)
        # incomplete measure from t=0 to t=1
        part.add(score.Measure(), 0, 1)

        # whole note from t=1 to t=5
        part.add(score.Note("c", 4), 1, 5)
        # add missing measures
        score.add_measures(part)
        time_pairs = [(-1, -1), (0, 0), (4, 4)]
        _test_time_pairs(part, time_pairs)

    def test_times_2(self):
        # 6/8 anacrusis
        part = score.Part("id")
        # 2 divs is 1 quarter
        part.set_quarter_duration(0, 2)
        part.add(score.TimeSignature(6, 8), 0)

        # ANACRUSIS
        part.add(score.Note("c", 4), 0, 3)
        part.add(score.Measure(), 0, 3)

        part.add(score.Note("c", 4), 3, 9)

        score.add_measures(part)

        time_pairs = [(-3, -1.5), (0, 0), (6, 3)]
        _test_time_pairs(part, time_pairs)


class TestSecondsToMidiTicks(unittest.TestCase):
    """
    Test seconds_to_midi_ticks with numpy array input -- the documented path
    that previously crashed because it used the removed np.int alias.
    """

    def test_array_input_returns_int_array(self):
        # The docstring promises an int-dtype array for array input; this is
        # the line that raised AttributeError on numpy >= 1.24.
        seconds = np.array([0.0, 0.5, 1.0, 2.0])
        ticks = seconds_to_midi_ticks(seconds)
        self.assertIsInstance(ticks, np.ndarray)
        self.assertTrue(np.issubdtype(ticks.dtype, np.integer))
        self.assertTrue(np.array_equal(ticks, np.array([0, 480, 960, 1920])))

    def test_array_matches_scalar(self):
        seconds = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
        ticks = seconds_to_midi_ticks(seconds)
        scalar_ticks = np.array([seconds_to_midi_ticks(float(s)) for s in seconds])
        self.assertTrue(np.array_equal(ticks, scalar_ticks))

    def test_array_roundtrip(self):
        seconds = np.array([0.0, 0.5, 1.0, 2.0])
        recovered = midi_ticks_to_seconds(seconds_to_midi_ticks(seconds))
        self.assertTrue(np.allclose(recovered, seconds))

    def test_array_non_default_mpq_ppq(self):
        seconds = np.array([0.0, 1.0, 2.0])
        ticks = seconds_to_midi_ticks(seconds, mpq=600000, ppq=960)
        scalar_ticks = np.array(
            [seconds_to_midi_ticks(float(s), mpq=600000, ppq=960) for s in seconds]
        )
        self.assertTrue(np.issubdtype(ticks.dtype, np.integer))
        self.assertTrue(np.array_equal(ticks, scalar_ticks))

    def test_empty_array(self):
        ticks = seconds_to_midi_ticks(np.array([]))
        self.assertIsInstance(ticks, np.ndarray)
        self.assertTrue(np.issubdtype(ticks.dtype, np.integer))
        self.assertEqual(len(ticks), 0)

    def test_scalar_still_returns_python_int(self):
        ticks = seconds_to_midi_ticks(1.0)
        self.assertIsInstance(ticks, int)
        self.assertEqual(ticks, 960)
