#!/usr/bin/env python
import numpy as np
import logging
import unittest

from partitura.utils.music import notes_to_pianoroll

LOGGER = logging.getLogger(__name__)

class TestPianoroll(unittest.TestCase):
    """
    Test piano roll from note array
    """

    def test_pianoroll(self):
        note_array = np.array([(60, 0, 1)],
                              dtype=[('pitch', 'i4'),
                                     ('onset', 'f4'),
                                     ('duration', 'f4')])
        
        pr = notes_to_pianoroll(note_array, pitch_margin=2, beat_div=2)
        expected_pr = np.array([[False, False],
                                [False, False],
                                [ True, False],
                                [False, False],
                                [False, False]])

        equal = np.all(pr.toarray() == expected_pr)
        
        self.assertEqual(equal, True)
