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

    def test_vosa(self):

        for fn in VOSA_TESTFILES:
            notearray = xml_to_notearray(fn)
            voices = estimate_voices(notearray)
            print(list(zip(notearray['pitch'], voices)))
        self.assertEqual(1, 1, f'jsjs')
