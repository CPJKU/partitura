import numpy as np

import unittest
from tempfile import TemporaryFile

from . import VOSA_TESTFILES

from partitura.importmusicxml import xml_to_notearray
from partitura.musicanalysis import estimate_voices


class TestVoSA(unittest.TestCase):
    """
    Test VoSA
    """

    def test_vosa_chew(self):
        # Example from Chew and Wu.
        notearray = xml_to_notearray(VOSA_TESTFILES[0])
        voices = estimate_voices(notearray, strictly_monophonic_voices=True)
        ground_truth_voices = np.array([3, 2, 1, 3, 3, 2, 3, 3, 2, 3,
                                        3, 1, 1, 2, 1, 1, 3, 2, 1, 1])
        self.assertTrue(np.all(voices == ground_truth_voices), 'Incorrect voice assignment.')
