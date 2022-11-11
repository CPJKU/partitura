#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for the `performance` module
"""
import unittest
import numpy as np

from partitura.performance import PerformedPart, Performance

RNG = np.random.RandomState(1984)


class TestPerformance(unittest.TestCase):
    def test_init_performance(self):

        note_arrays = [generate_random_note_array(100) for i in range(3)]

        performedparts = [PerformedPart.from_note_array(na) for na in note_arrays]

        perf_from_ppart = Performance(
            id="",
            performedparts=performedparts[0],
        )

        self.assertTrue(isinstance(perf_from_ppart, Performance))
        perf_from_ppart_list = Performance(id="", performedparts=performedparts)
        self.assertTrue(isinstance(perf_from_ppart_list, Performance))

        try:

            class NotAPerformance:
                pass

            other = NotAPerformance()
            perf_from_other = Performance(id="", performedparts=other)

        except ValueError:
            # assert that other is not a performed part
            self.assertTrue(not isinstance(other, PerformedPart))

    def test_num_tracks(self):

        num_parts_tracks = RNG.randint(3, 10, (10, 2))

        for nparts, ntracks in num_parts_tracks:
            # all of these arrays have tracks numbered 0-ntracks - 1
            note_arrays = [
                generate_random_note_array(100, n_tracks=ntracks) for i in range(nparts)
            ]

            performedparts = [PerformedPart.from_note_array(na) for na in note_arrays]

            # test that the number of tracks within each performed part is correct
            self.assertTrue(all([pp.num_tracks == ntracks for pp in performedparts]))

            performance = Performance(
                id="", performedparts=performedparts, ensure_unique_tracks=False
            )

            # test whether the number of parts is correct
            self.assertEqual(performance.num_tracks, nparts * ntracks)

    def test_sanitize_track_numbers(self):

        num_parts_tracks = RNG.randint(3, 10, (10, 2))

        for nparts, ntracks in num_parts_tracks:

            note_arrays = [
                generate_random_note_array(100, n_tracks=ntracks) for i in range(nparts)
            ]

            performedparts = [PerformedPart.from_note_array(na) for na in note_arrays]

            performance = Performance(
                id="", performedparts=performedparts, ensure_unique_tracks=False
            )

            note_array_before = performance.note_array()

            tracks = np.unique(note_array_before["track"])

            self.assertTrue(np.all(tracks == np.arange(ntracks)))

            # sanitize track numbers
            performance.sanitize_track_numbers()

            note_array_after = performance.note_array()

            after_tracks = np.unique(note_array_after["track"])

            self.assertTrue(np.all(after_tracks == np.arange(ntracks * nparts)))


def generate_random_note_array(n_notes=100, beat_period=0.5, n_tracks=3):

    note_array = np.empty(
        (n_notes),
        dtype=[
            ("onset_sec", "f4"),
            ("duration_sec", "f4"),
            ("pitch", "i4"),
            ("velocity", "i4"),
            ("track", "i4"),
            ("channel", "i4"),
            ("id", "U256"),
        ],
    )

    note_array["pitch"] = RNG.randint(0, 128, n_notes)

    ioi = RNG.rand(n_notes - 1) * beat_period

    note_array["onset_sec"] = np.r_[0, np.cumsum(ioi)]

    note_array["duration_sec"] = np.clip(
        RNG.rand(n_notes) * 2 * beat_period,
        a_min=0.3,
        a_max=2,
    )

    note_array["velocity"] = RNG.randint(54, 90, n_notes)

    note_array["channel"] *= 0

    note_array["id"] = np.array([f"n{i}" for i in range(n_notes)])

    track_idxs = np.arange(n_notes)
    RNG.shuffle(track_idxs)

    track_length = int(np.floor(n_notes / n_tracks))

    for i in range(n_tracks):
        if i < n_tracks - 1:
            idx = track_idxs[i * track_length : (i + 1) * track_length]
        else:
            idx = track_idxs[i * track_length :]
        note_array["track"][idx] = i
    return note_array
