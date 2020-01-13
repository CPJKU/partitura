#!/usr/bin/env python
import numpy as np
import logging
import unittest

from partitura.utils.music import notearray_to_pianoroll, pianoroll_to_notearray

LOGGER = logging.getLogger(__name__)


class TestPianorollFromNotes(unittest.TestCase):
    """
    Test piano roll from note array
    """

    def test_score_pianoroll(self):
        note_array = np.array([(60, 0, 1)],
                              dtype=[('pitch', 'i4'),
                                     ('onset', 'f4'),
                                     ('duration', 'f4')])

        pr = notearray_to_pianoroll(note_array, pitch_margin=2, time_div=2)
        expected_pr = np.array([[0, 0],
                                [0, 0],
                                [1, 1],
                                [0, 0],
                                [0, 0]])

        equal = np.all(pr.toarray() == expected_pr)

        self.assertEqual(equal, True)

    def test_performance_pianoroll(self):
        note_array = np.array([(60, 0, 1, 72)],
                              dtype=[('pitch', 'i4'),
                                     ('p_onset', 'f4'),
                                     ('p_duration', 'f4'),
                                     ('velocity', 'i4')])

        pr = notearray_to_pianoroll(note_array, pitch_margin=2, time_div=2,
                                    is_performance=True)
        expected_pr = np.array([[0, 0],
                                [0, 0],
                                [72, 72],
                                [0, 0],
                                [0, 0]])

        equal = np.all(pr.toarray() == expected_pr)

        self.assertEqual(equal, True)


class TestNotesFromPianoroll(unittest.TestCase):
    """
    Test piano roll from note array
    """

    def test_pianoroll_to_notearray(self):
        time_div = 8
        note_array = np.array([(60, 0, 2, 40),
                               (65, 0, 1, 15),
                               (67, 0, 1, 72),
                               (69, 1, 1, 90),
                               (66, 2, 1, 80)],
                              dtype=[('pitch', 'i4'),
                                     ('p_onset', 'f4'),
                                     ('p_duration', 'f4'),
                                     ('velocity', 'i4')])

        pr = notearray_to_pianoroll(note_array,
                                    time_div=time_div,
                                    note_separation=False,
                                    is_performance=True)

        rec_note_array = pianoroll_to_notearray(pr, time_div)

        # sort by onset and pitch
        original_pitch_idx = np.argsort(note_array['pitch'])
        note_array = note_array[original_pitch_idx]
        original_onset_idx = np.argsort(note_array['p_onset'], kind='mergesort')
        note_array = note_array[original_onset_idx]

        rec_pitch_idx = np.argsort(rec_note_array['pitch'])
        rec_note_array = rec_note_array[rec_pitch_idx]
        rec_onset_idx = np.argsort(rec_note_array['p_onset'], kind='mergesort')
        rec_note_array = rec_note_array[rec_onset_idx]

        test = np.all(note_array == rec_note_array)
        self.assertEqual(test, True)
