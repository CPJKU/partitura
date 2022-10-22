#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the utility methods.
"""
import unittest
import partitura
import numpy as np

from partitura.utils import music
from tests import MATCH_IMPORT_EXPORT_TESTFILES, VOSA_TESTFILES

RNG = np.random.RandomState(1984)


class TestGetMatchedNotes(unittest.TestCase):
    def test_get_matched_notes(self):
        for fn in MATCH_IMPORT_EXPORT_TESTFILES:
            perf, alignment, scr = partitura.load_match(
                filename=fn,
                create_score=True,
            )
            perf_note_array = perf.note_array()
            scr_note_array = scr.note_array()
            matched_idxs = music.get_matched_notes(
                spart_note_array=scr_note_array,
                ppart_note_array=perf_note_array,
                alignment=alignment,
            )
            scr_pitch = scr_note_array["pitch"][matched_idxs[:, 0]]
            perf_pitch = perf_note_array["pitch"][matched_idxs[:, 1]]

            self.assertTrue(np.all(scr_pitch == perf_pitch))


class TestGetTimeMapsFromAlignment(unittest.TestCase):
    def test_get_time_maps_from_alignment(self):
        for fn in VOSA_TESTFILES:
            scr = partitura.load_musicxml(fn)
            note_ids = scr.note_array()["id"]
            beats_per_minute = 60 / RNG.uniform(0.3, 3, size=2)

            for bpm in beats_per_minute:
                ppart = music.performance_from_part(part=scr[0], bpm=bpm)
                alignment = [
                    dict(label="match", score_id=sid, performance_id=sid)
                    for sid in note_ids
                ]

                (
                    ptime_to_stime_map,
                    stime_to_ptime_map,
                ) = music.get_time_maps_from_alignment(
                    spart_or_note_array=scr[0],
                    ppart_or_note_array=ppart,
                    alignment=alignment,
                    remove_ornaments=True,
                )

                score_onsets = np.arange(4, 0.5)
                performed_onsets = 60 / bpm * score_onsets

                self.assertTrue(
                    np.all(score_onsets == ptime_to_stime_map(performed_onsets))
                )
                self.assertTrue(
                    np.all(performed_onsets == stime_to_ptime_map(score_onsets))
                )


class TestPerformanceFromPart(unittest.TestCase):
    def test_performance_from_part(self):
        for fn in VOSA_TESTFILES:
            scr = partitura.load_musicxml(fn)
            beats_per_minute = 60 / RNG.uniform(0.3, 3, size=2)
            midi_velocity = RNG.randint(30, 127, size=2)
            for bpm, vel in zip(beats_per_minute, midi_velocity):
                ppart = music.performance_from_part(part=scr[0], bpm=bpm, velocity=vel)

                # assert that both objects have the same number of notes
                self.assertEqual(len(scr[0].notes_tied), len(ppart.notes))

                snote_array = scr[0].note_array()
                pnote_array = ppart.note_array()

                # check MIDI velocities
                self.assertTrue(np.all(pnote_array["velocity"] == vel))

                alignment = [
                    dict(label="match", score_id=sid, performance_id=sid)
                    for sid in snote_array["id"]
                ]

                matched_idxs = music.get_matched_notes(
                    spart_note_array=snote_array,
                    ppart_note_array=pnote_array,
                    alignment=alignment,
                )

                # check pitch
                self.assertTrue(
                    np.all(
                        pnote_array["pitch"][matched_idxs[:, 1]]
                        == snote_array["pitch"][matched_idxs[:, 0]]
                    )
                )
                # check durations
                self.assertTrue(
                    np.allclose(
                        pnote_array["duration_sec"],
                        snote_array["duration_beat"] * (60 / bpm),
                    )
                )

                pnote_array = pnote_array[matched_idxs[:, 1]]
                snote_array = snote_array[matched_idxs[:, 0]]

                unique_onsets = np.unique(snote_array["onset_beat"])
                unique_onset_idxs = [
                    np.where(snote_array["onset_beat"] == uo)[0] for uo in unique_onsets
                ]

                # check performed onsets
                perf_onsets = np.array(
                    [
                        np.mean(pnote_array["onset_sec"][uix])
                        for uix in unique_onset_idxs
                    ]
                )

                beat_period = np.diff(perf_onsets) / np.diff(unique_onsets)

                # check that that the performance corresponds to the expected tempo
                self.assertTrue(np.allclose(60 / beat_period, bpm))
