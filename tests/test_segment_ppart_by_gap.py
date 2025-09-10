import numpy as np
import unittest

from partitura.performance import PerformedPart
from partitura.utils.music import segment_ppart_by_gap

def make_ppart(note_times, note_durations):
        notes = []
        for i, (on, dur) in enumerate(zip(note_times, note_durations)):
            notes.append({
                "note_on": on,
                "note_off": on + dur,
                "sound_off": on + dur,
                "id": f"n{i}",
                "pitch": 60,
                "velocity": 64,
                "track": 0,
                "channel": 0,
            })
        return PerformedPart(notes=notes)

class TestSegmentPpartByGap(unittest.TestCase):
    def test_no_gaps_returns_none(self):
        ppart = make_ppart([0, 1, 2, 3], [1, 1, 1, 1])
        segments, times = segment_ppart_by_gap(ppart, gap=10)
        self.assertTrue(segments is None and times is None)

    def test_single_gap_splits_in_two(self):
        ppart = make_ppart([0, 1, 6, 11, 25, 26, 31, 35], [1, 1, 1, 1, 1, 1, 1, 1])
        segments, times = segment_ppart_by_gap(ppart, gap=10)
        self.assertEqual(len(segments), 2)
        self.assertEqual(times[0][0], 0)
        self.assertEqual(times[0][1], 12)
        self.assertEqual(times[1][0], 25)
        self.assertEqual(times[1][1], 36)

    def test_use_sound_off(self):
        ppart = make_ppart([0, 1, 6, 11, 25, 26, 31, 35], [1, 1, 1, 1, 1, 1, 1, 1])
        # artificially set sound_off to create a gap
        ppart.notes[3]["sound_off"] = 24
        segments, times = segment_ppart_by_gap(ppart, gap=10, use_sound_off=True)
        self.assertTrue(segments is None and times is None)
        segments, times = segment_ppart_by_gap(ppart, gap=10, use_sound_off=False)
        self.assertEqual(len(segments), 2)
        self.assertEqual(len(times), 2)
        self.assertEqual(times[0][0], 0)
        self.assertEqual(times[0][1], 12)
        self.assertEqual(times[1][0], 25)
        self.assertEqual(times[1][1], 36)

    def test_min_segment_duration_merges_short_segments(self):
        ppart = make_ppart([0, 1, 13, 14], [1, 1, 1, 1])
        segments, times = segment_ppart_by_gap(ppart, gap=10, min_segment_duration=5)
        # Should merge the short segments into one
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(times), 1)
        self.assertEqual(times[0][0], 0)
        self.assertEqual(times[0][1], 15)

    def test_gap_at_end(self):
        ppart = make_ppart([0, 1, 2, 7, 11, 25], [1, 1, 1, 1, 1, 1])
        segments, times = segment_ppart_by_gap(ppart, gap=10)
        # Should merge last segment with the previous one
        self.assertEqual(len(segments), 1)
        self.assertEqual(times[0][0], 0)
        self.assertEqual(times[0][1], 26)