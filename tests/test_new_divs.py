#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for methods for handling divisions and time signatures.
"""
import logging
import unittest

import partitura.score as score

LOGGER = logging.getLogger(__name__)


def create_part_from_spec(spec):
    """Create a part from a specification of divisions, time signatures
    and notes.

    This is a helper function for the TestBeatMap test case.

    Parameters
    ----------
    spec : dictionary
        Part specification

    Returns
    -------
    Part
        Part instance

    """

    part = score.Part("beatmaptest")
    # for t, divs in spec['divs']:
    #     part.add(score.Divisions(divs), t)
    for t, divs in spec["divs"]:
        part.set_quarter_duration(t, divs)

    for t, num, den in spec["ts"]:
        part.add(score.TimeSignature(num, den), t)

    # divs_map = part.divisions_map

    for t, dur in spec["notes"]:
        # sd = score.estimate_symbolic_duration(dur, int(divs_map(t)))
        part.add(score.Note(step="A", alter=None, octave=4), t, t + dur)

    score.add_measures(part)

    return part


class TestBeatMap(unittest.TestCase):
    """Test that divisions and time signatures are handled correctly
    when computing beat times and symbolic durations.

    Each of the test cases specifies the contents of a part in terms
    of divisions, time signatures, and notes, and corresponding
    targets in terms of the symbolic durations of each note and their
    onset times in beats.

    For each test case, the part is constructed according to the
    specification, and the computed symbolic durations and onsets are
    compared to the targets. Furthermore the beat map (mapping
    divisions times to beat times) and inverse beat map (mapping beat
    times to divisions times) are jointly tested for their
    invertibility.

    """

    test_cases = [
        {
            "part_spec": {
                "divs": ((0, 10), (50, 2)),  # at time=0, divs=10  # at time=40, divs=2
                "ts": (
                    (0, 4, 4),  # at time=0, ts=4/4
                    (40, 3, 8),
                ),  # at time=40, ts=3/8
                "notes": (
                    (0, 10),  # at time=0, dur=10
                    (10, 10),
                    (20, 10),
                    (30, 10),
                    (40, 5),
                    (45, 5),
                    (50, 1),
                    (51, 3),
                ),
            },
            "target": {
                "sym_durs": (
                    "quarter",
                    "quarter",
                    "quarter",
                    "quarter",
                    "eighth",
                    "eighth",
                    "eighth",
                    "quarter.",
                ),
                "onset_beats": (0, 1, 2, 3, 4, 5, 6, 7),
            },
        },
        {
            "part_spec": {
                "divs": (
                    (0, 10),  # at time=0, divs=10
                    (45, 2),  # at time=40, divs=2
                    (46, 10),
                ),  # at time=47, divs=10
                "ts": (
                    (0, 4, 4),  # at time=0, ts=4/4
                    (40, 3, 8),
                ),  # at time=40, ts=3/8
                "notes": (
                    (0, 10),  # at time=0, dur=10
                    (10, 10),
                    (20, 10),
                    (30, 10),
                    (40, 5),
                    (45, 1),
                    (46, 5),
                    (51, 15),
                ),
            },
            "target": {
                "sym_durs": (
                    "quarter",
                    "quarter",
                    "quarter",
                    "quarter",
                    "eighth",
                    "eighth",
                    "eighth",
                    "quarter.",
                ),
                "onset_beats": (0, 1, 2, 3, 4, 5, 6, 7),
            },
        },
        {
            "part_spec": {
                "divs": ((0, 10),),
                "ts": (
                    (0, 4, 4),  # at time=0, ts=4/4
                    (10, 3, 4),
                ),  # at time=10, ts=3/4
                "notes": ((0, 10), (10, 10), (20, 10), (30, 10)),  # at time=0, dur=10
            },
            "target": {
                "sym_durs": ("quarter", "quarter", "quarter", "quarter"),
                "onset_beats": (-1, 0, 1, 2),
            },
        },
    ]

    def test_beat_map_cases(self):
        for test_case in self.test_cases:
            part = create_part_from_spec(test_case["part_spec"])
            self._test_symbolic_durations(part, test_case["target"]["sym_durs"])
            self._test_note_onsets(part, test_case["target"]["onset_beats"])
            self._test_beat_map(part)

    def _test_symbolic_durations(self, part, target_durations):
        notes = list(part.iter_all(score.Note))
        if len(notes) != len(target_durations):
            LOGGER.warning(
                "Skipping incorrect test case (input and targets do not match)"
            )
            return

        for target_sd, note in zip(target_durations, notes):
            est_sd = score.format_symbolic_duration(note.symbolic_duration)
            msg = "{} != {} ({})".format(target_sd, est_sd, note.start.t)
            self.assertEqual(target_sd, est_sd, msg)

    def _test_note_onsets(self, part, target_onsets):
        notes = list(part.iter_all(score.Note))
        if len(notes) != len(target_onsets):
            LOGGER.warning(
                "Skipping incorrect test case (input and targets do not match)"
            )
            return
        beat_map = part.beat_map
        for target_onset, note in zip(target_onsets, notes):
            self.assertAlmostEqual(float(target_onset), beat_map(note.start.t))

    def _test_beat_map(self, part):
        beat_map = part.beat_map
        inv_beat_map = part.inv_beat_map
        test_times = range(part.first_point.t, part.last_point.t)
        for t in test_times:
            self.assertAlmostEqual(t, inv_beat_map(beat_map(t)))


if __name__ == "__main__":
    unittest.main()
