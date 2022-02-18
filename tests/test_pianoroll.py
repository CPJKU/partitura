#!/usr/bin/env python
import numpy as np
import logging
import unittest

from partitura.utils.music import compute_pianoroll, pianoroll_to_notearray
from partitura import load_musicxml

from tests import MUSICXML_IMPORT_EXPORT_TESTFILES

LOGGER = logging.getLogger(__name__)


class TestPianorollFromNotes(unittest.TestCase):
    """
    Test piano roll from note array
    """

    def test_score_pianoroll(self):
        note_array = np.array(
            [(60, 0, 1)],
            dtype=[("pitch", "i4"), ("onset_beat", "f4"), ("duration_beat", "f4")],
        )

        pr = compute_pianoroll(note_array, pitch_margin=2, time_div=2)
        expected_pr = np.array([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]])

        equal = np.all(pr.toarray() == expected_pr)

        self.assertEqual(equal, True)

    def test_performance_pianoroll(self):
        note_array = np.array(
            [(60, 0, 1, 72)],
            dtype=[
                ("pitch", "i4"),
                ("onset_sec", "f4"),
                ("duration_sec", "f4"),
                ("velocity", "i4"),
            ],
        )

        pr = compute_pianoroll(note_array, pitch_margin=2, time_div=2)
        expected_pr = np.array([[0, 0], [0, 0], [72, 72], [0, 0], [0, 0]])

        equal = np.all(pr.toarray() == expected_pr)

        self.assertTrue(equal)

    def test_performance_pianoroll_onset_only(self):
        note_array = np.array(
            [(60, 0, 1, 72)],
            dtype=[
                ("pitch", "i4"),
                ("onset_sec", "f4"),
                ("duration_sec", "f4"),
                ("velocity", "i4"),
            ],
        )

        pr = compute_pianoroll(note_array, pitch_margin=3, time_div=2, onset_only=True)
        expected_pr = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [72, 0],
                [0, 0],
                [0, 0],
                [0, 0]
            ]
        )

        equal = np.all(pr.toarray() == expected_pr)

        self.assertTrue(equal)

    def test_noteduration_pianoroll(self):
        note_array = np.array(
            [(60, 0, 2),
             (60, 2, 2),
             (60, 5, 0.3)],
            dtype=[("pitch", "i4"), ("onset_beat", "f4"), ("duration_beat", "f4")],
        )

        pr = compute_pianoroll(note_array, pitch_margin=2, time_div=1, onset_only=True)

        expected_pr = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ]
        )

        equal = np.all(pr.toarray() == expected_pr)
        self.assertTrue(equal)

    def test_time_margin_pianoroll(self):
        note_array = np.array(
            [(60, 0, 2),
             (60, 2, 2),
             (60, 5, 0.3)],
            dtype=[("pitch", "i4"), ("onset_beat", "f4"), ("duration_beat", "f4")],
        )

        for tm in range(10):
            pr = compute_pianoroll(note_array, pitch_margin=2, time_div=1,
                                   time_margin=tm, onset_only=True)

            expected_pr = np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ]
            )

            time_margins = np.zeros((5, tm))
            expected_pr = np.column_stack((time_margins, expected_pr, time_margins))
            equal = np.all(pr.toarray() == expected_pr)
            self.assertTrue(equal)


class TestNotesFromPianoroll(unittest.TestCase):
    """
    Test piano roll from note array
    """

    def test_pianoroll_to_notearray(self):
        time_div = 8
        note_array = np.array(
            [
                (60, 0, 2, 40, "n0"),
                (65, 0, 1, 15, "n1"),
                (67, 0, 1, 72, "n2"),
                (69, 1, 1, 90, "n3"),
                (66, 2, 1, 80, "n4"),
            ],
            dtype=[
                ("pitch", "i4"),
                ("onset_sec", "f4"),
                ("duration_sec", "f4"),
                ("velocity", "i4"),
                ("id", "U256")
            ],
        )

        pr = compute_pianoroll(note_array, time_div=time_div, note_separation=False)

        rec_note_array = pianoroll_to_notearray(pr, time_div)

        # sort by onset and pitch
        original_pitch_idx = np.argsort(note_array["pitch"])
        note_array = note_array[original_pitch_idx]
        original_onset_idx = np.argsort(note_array["onset_sec"], kind="mergesort")
        note_array = note_array[original_onset_idx]

        rec_pitch_idx = np.argsort(rec_note_array["pitch"])
        rec_note_array = rec_note_array[rec_pitch_idx]
        rec_onset_idx = np.argsort(rec_note_array["onset_sec"], kind="mergesort")
        rec_note_array = rec_note_array[rec_onset_idx]

        test = np.all(note_array == rec_note_array)
        self.assertTrue(test)

    def test_reconstruction_score(self):

        for fn in MUSICXML_IMPORT_EXPORT_TESTFILES:
            spart = load_musicxml(fn)
            note_array = spart.note_array
            pr = compute_pianoroll(spart, time_unit="div", time_div=1, remove_silence=False)

            rec_note_array = pianoroll_to_notearray(pr, time_div=1, time_unit="div")

            original_pitch_idx = np.argsort(note_array["pitch"])
            note_array = note_array[original_pitch_idx]
            original_onset_idx = np.argsort(note_array["onset_div"], kind="mergesort")
            note_array = note_array[original_onset_idx]

            rec_pitch_idx = np.argsort(rec_note_array["pitch"])
            rec_note_array = rec_note_array[rec_pitch_idx]
            rec_onset_idx = np.argsort(rec_note_array["onset_div"], kind="mergesort")
            rec_note_array = rec_note_array[rec_onset_idx]

            test_pitch = np.all(note_array['pitch'] == rec_note_array['pitch'])
            self.assertTrue(test_pitch)
            test_onset = np.all(note_array['onset_div'] == rec_note_array['onset_div'])
            self.assertTrue(test_onset)
            test_duration = np.all(note_array['duration_div'] == rec_note_array['duration_div'])
            self.assertTrue(test_duration)

    def test_reconstruction_perf(self):

        rng = np.random.RandomState(1984)
        piece_length = 11
        for i in range(10):

            note_array = np.zeros(
                piece_length,
                dtype=[
                    ("pitch", "i4"),
                    ("onset_sec", "f4"),
                    ("duration_sec", "f4"),
                    ("velocity", "i4"),
                    ("id", "U256")
                ]
            )

            note_array['pitch'] = rng.randint(0, 127, piece_length)
            note_array['duration_sec'] = np.clip(
                np.round(rng.rand(piece_length) * 2, 2),
                a_max=None,
                a_min=0.01
            )

            onset = np.round(np.r_[0, np.cumsum(note_array['duration_sec'] + 0.02)], 2)
            note_array['onset_sec'] = onset[:-1]
            note_array['velocity'] = rng.randint(20, 127, piece_length)
            note_array['id'] = np.array([f'n{nid}' for nid in range(piece_length)])

            pr = compute_pianoroll(note_array, time_unit="sec", time_div=100,
                                   remove_silence=False)

            rec_note_array = pianoroll_to_notearray(pr, time_div=100, time_unit="sec")
            rec_pr = compute_pianoroll(rec_note_array, time_unit="sec", time_div=100,
                                       remove_silence=False)

            # assert piano rolls are the same
            self.assertTrue(np.all(rec_pr.toarray() == pr.toarray()))

            # assert note arrays are the same
            test_pitch = np.all(note_array['pitch'] == rec_note_array['pitch'])
            self.assertTrue(test_pitch)
            test_onset = np.all(note_array['onset_sec'] == rec_note_array['onset_sec'])
            self.assertTrue(test_onset)
            test_duration = np.all(note_array['duration_sec'] == rec_note_array['duration_sec'])
            self.assertTrue(test_duration)
            test_velocity = np.all(note_array['velocity'] == rec_note_array['velocity'])
            self.assertTrue(test_velocity)
