#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for methods for computing piano rolls.
"""
import numpy as np
import logging
import unittest
from functools import partial

from partitura.utils.music import (
    compute_pianoroll,
    pianoroll_to_notearray,
    compute_pitch_class_pianoroll,
)
from partitura import load_musicxml, load_score, load_performance
import partitura

from tests import (
    MUSICXML_IMPORT_EXPORT_TESTFILES,
    PIANOROLL_TESTFILES,
    KERN_TESTFILES,
    MOZART_VARIATION_FILES,
)

LOGGER = logging.getLogger(__name__)

RNG = np.random.RandomState(1984)


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
            [[0, 0], [0, 0], [0, 0], [72, 0], [0, 0], [0, 0], [0, 0]]
        )

        equal = np.all(pr.toarray() == expected_pr)

        self.assertTrue(equal)

    def test_noteduration_pianoroll(self):
        note_array = np.array(
            [(60, 0, 2), (60, 2, 2), (60, 5, 0.3)],
            dtype=[("pitch", "i4"), ("onset_beat", "f4"), ("duration_beat", "f4")],
        )

        pr = compute_pianoroll(note_array, pitch_margin=2, time_div=1, onset_only=True)

        expected_pr = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        equal = np.all(pr.toarray() == expected_pr)
        self.assertTrue(equal)

    def test_time_margin_pianoroll(self):
        note_array = np.array(
            [(60, 0, 2), (60, 2, 2), (60, 5, 0.3)],
            dtype=[("pitch", "i4"), ("onset_beat", "f4"), ("duration_beat", "f4")],
        )

        for tm in range(10):
            pr = compute_pianoroll(
                note_array, pitch_margin=2, time_div=1, time_margin=tm, onset_only=True
            )

            expected_pr = np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )

            time_margins = np.zeros((5, tm))
            expected_pr = np.column_stack((time_margins, expected_pr, time_margins))
            equal = np.all(pr.toarray() == expected_pr)
            self.assertTrue(equal)

    def test_binary_pianoroll(self):
        """
        Test `binary` parameter in `compute_pianoroll`.
        """
        # Test with a performance since they include MIDI velocity
        # in the piano roll.
        performance_fn = MOZART_VARIATION_FILES["midi"]

        performance = load_performance(performance_fn)

        note_array = performance.note_array()

        piano_roll_non_binary, idx_non_binary = compute_pianoroll(
            note_info=performance, binary=False, return_idxs=True
        )

        piano_roll_binary, idx_binary = compute_pianoroll(
            note_info=performance, binary=True, return_idxs=True
        )

        # assert that the maximal value of the binary piano roll is 1
        self.assertTrue(piano_roll_binary.max() == 1)
        # assert that the opposite is true for the non_binary piano roll
        # (this is only the case for performances where there is MIDI velocity)
        self.assertTrue(piano_roll_non_binary.max() == note_array["velocity"].max())

        # assert that indices in both piano rolls are the same
        self.assertTrue(np.all(idx_non_binary == idx_binary))

        # Test that the binary piano roll has only values in 0 and one
        unique_values_binary = np.unique(piano_roll_binary.toarray())
        self.assertTrue(set(unique_values_binary) == set([0, 1]))

        # Assert that the binary piano roll is equivalent to binarizing
        # the original piano roll
        binarized_pr = piano_roll_non_binary.toarray().copy()
        binarized_pr[binarized_pr != 0] = 1
        self.assertTrue(np.all(binarized_pr == piano_roll_binary.toarray()))


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
                ("id", "U256"),
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
            score = load_musicxml(fn)
            note_array = score[0].note_array()
            pr = compute_pianoroll(
                score[0], time_unit="div", time_div=1, remove_silence=False
            )
            rec_note_array = pianoroll_to_notearray(pr, time_div=1, time_unit="div")

            original_pitch_idx = np.argsort(note_array["pitch"])
            note_array = note_array[original_pitch_idx]
            original_onset_idx = np.argsort(note_array["onset_div"], kind="mergesort")
            note_array = note_array[original_onset_idx]

            rec_pitch_idx = np.argsort(rec_note_array["pitch"])
            rec_note_array = rec_note_array[rec_pitch_idx]
            rec_onset_idx = np.argsort(rec_note_array["onset_div"], kind="mergesort")
            rec_note_array = rec_note_array[rec_onset_idx]
            
            test_pitch = np.all(note_array["pitch"] == rec_note_array["pitch"])
            self.assertTrue(test_pitch)
            test_onset = np.all(note_array["onset_div"] == rec_note_array["onset_div"])
            self.assertTrue(test_onset)
            test_duration = np.all(
                note_array["duration_div"] == rec_note_array["duration_div"]
            )
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
                    ("id", "U256"),
                ],
            )

            note_array["pitch"] = rng.randint(0, 127, piece_length)
            note_array["duration_sec"] = np.clip(
                np.round(rng.rand(piece_length) * 2, 2), a_max=None, a_min=0.01
            )

            onset = np.round(np.r_[0, np.cumsum(note_array["duration_sec"] + 0.02)], 2)
            note_array["onset_sec"] = onset[:-1]
            note_array["velocity"] = rng.randint(20, 127, piece_length)
            note_array["id"] = np.array([f"n{nid}" for nid in range(piece_length)])

            pr = compute_pianoroll(
                note_array, time_unit="sec", time_div=100, remove_silence=False
            )

            rec_note_array = pianoroll_to_notearray(pr, time_div=100, time_unit="sec")
            rec_pr = compute_pianoroll(
                rec_note_array, time_unit="sec", time_div=100, remove_silence=False
            )

            # assert piano rolls are the same
            self.assertTrue(np.all(rec_pr.toarray() == pr.toarray()))

            # assert note arrays are the same
            test_pitch = np.all(note_array["pitch"] == rec_note_array["pitch"])
            self.assertTrue(test_pitch)
            test_onset = np.all(note_array["onset_sec"] == rec_note_array["onset_sec"])
            self.assertTrue(test_onset)
            test_duration = np.all(
                note_array["duration_sec"] == rec_note_array["duration_sec"]
            )
            self.assertTrue(test_duration)
            test_velocity = np.all(note_array["velocity"] == rec_note_array["velocity"])
            self.assertTrue(test_velocity)


class TestPianorollFromScores(unittest.TestCase):
    """
    Test piano roll from scores
    """

    def test_score_pianoroll(self):
        # normally call the function
        parts = load_score(PIANOROLL_TESTFILES[0])
        pr0 = compute_pianoroll(parts[0])
        pr1 = compute_pianoroll(parts[1])
        pr2 = compute_pianoroll(parts[2])
        self.assertTrue(pr0.shape != pr1.shape)
        self.assertTrue(pr1.shape != pr2.shape)
        # remove the silence
        parts = load_score(PIANOROLL_TESTFILES[0])
        pr0 = compute_pianoroll(
            parts[0], time_unit="beat", time_div=1, remove_silence=False
        )
        pr1 = compute_pianoroll(
            parts[1], time_unit="beat", time_div=1, remove_silence=False
        )
        pr2 = compute_pianoroll(
            parts[2], time_unit="beat", time_div=1, remove_silence=False
        )
        self.assertTrue(pr0.shape == (128, 12))
        self.assertTrue(pr1.shape == (128, 8))
        self.assertTrue(pr0.shape == (128, 12))
        # set a fixed end
        parts = load_score(PIANOROLL_TESTFILES[0])
        pr0 = compute_pianoroll(
            parts[0], time_unit="beat", time_div=2, remove_silence=False
        )
        pr1 = compute_pianoroll(
            parts[1], time_unit="beat", time_div=2, remove_silence=False, end_time=12
        )
        pr2 = compute_pianoroll(
            parts[2], time_unit="beat", time_div=2, remove_silence=False
        )
        self.assertTrue(pr0.shape == (128, 24))
        self.assertTrue(pr1.shape == (128, 24))
        self.assertTrue(pr0.shape == (128, 24))

    def test_sum_pianoroll(self):
        time_div = 4
        parts = load_score(PIANOROLL_TESTFILES[2])
        prs = []
        for part in parts:
            prs.append(compute_pianoroll(part, time_unit="beat", time_div=time_div))
        pianoroll_sum = prs[0] + prs[1] + prs[2] + prs[3]
        original_pianoroll = compute_pianoroll(
            parts, time_unit="beat", time_div=time_div
        ).toarray()
        self.assertTrue(pianoroll_sum.shape == original_pianoroll.shape)
        clipped_pr_sum = np.clip(
            pianoroll_sum.toarray(), 0, 1
        )  # remove count for double notes
        self.assertTrue(np.array_equal(clipped_pr_sum, original_pianoroll))

    def test_pianoroll_length(self):
        score = load_score(KERN_TESTFILES[7])
        parts = score.parts
        # parts = list(partitura.score.iter_parts(score))
        # set musical beat if requested
        for part in parts:
            part.use_musical_beat()
        # get the maximum length of all parts to avoid shorter pianorolls
        end_time = max([part.beat_map([part._points[-1].t]) for part in parts])
        # define the parameters of the compute_pianoroll function
        get_pianoroll = partial(
            partitura.utils.compute_pianoroll,
            time_unit="beat",
            time_div=12,
            piano_range=True,
            remove_silence=False,
            end_time=end_time,
        )
        # compute pianorolls for all separated voices
        prs = [get_pianoroll(part) for part in parts]
        self.assertTrue(pr.shape == prs[0].shape for pr in prs)


class TestPitchClassPianoroll(unittest.TestCase):
    """
    Test pitch class piano roll
    """

    def test_midi_pitch_to_pitch_class(self):
        """
        Test that all MIDI pitches would be correctly represented
        in the pitch class piano roll
        """
        for pitch in range(128):
            note_array = np.array(
                [(pitch, 0, 1)],
                dtype=[
                    ("pitch", "i4"),
                    ("onset_beat", "f4"),
                    ("duration_beat", "f4"),
                ],
            )

            time_div = 2
            pr = compute_pitch_class_pianoroll(note_array, time_div=time_div)

            expected_pr = np.zeros((12, time_div))

            expected_pr[pitch % 12] = 1

            equal = np.all(pr == expected_pr)

            self.assertEqual(equal, True)

    def test_indices(self):
        """
        Test indices from the piano roll
        """
        # Generate a random piano roll
        note_array = partitura.utils.music.generate_random_performance_note_array(100)
        pianoroll, pr_idxs = compute_pianoroll(
            note_info=note_array,
            return_idxs=True,
            time_unit="sec",
            time_div=10,
        )

        pc_pianoroll, pcr_idxs = compute_pitch_class_pianoroll(
            note_info=note_array,
            return_idxs=True,
            time_unit="sec",
            time_div=10,
        )

        # Assert that there is an index for each note
        self.assertTrue(len(pcr_idxs) == len(note_array))
        self.assertTrue(len(pcr_idxs) == len(pr_idxs))

        # Assert that the indices correspond to the same notes as in the piano roll
        self.assertTrue(np.all(pcr_idxs[:, 3] == note_array["pitch"]))

        # Test that MIDI pitch and pitch class are correct
        self.assertTrue(np.all(np.mod(pr_idxs[:, 3], 12) == pcr_idxs[:, 0]))
        # Assert that MIDI pitch info is identical for pc_pianoroll and
        # regular piano rolls
        self.assertTrue(np.all(pr_idxs[:, 3] == pcr_idxs[:, 3]))

        # Onsets and offsets should be identical
        self.assertTrue(np.all(pr_idxs[:, 2:4] == pcr_idxs[:, 2:4]))

if __name__ == "__main__":
    unittest.main()
