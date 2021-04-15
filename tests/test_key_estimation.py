import unittest

from partitura import EXAMPLE_MUSICXML
from partitura import load_musicxml
from partitura.musicanalysis import estimate_key


class TestKeyEstimation(unittest.TestCase):
    """
    Test key estimation
    """

    score = load_musicxml(EXAMPLE_MUSICXML)

    def test_part(self):
        key = estimate_key(self.score)
        self.assertTrue(key == "Am", "Incorrect key")

    def test_note_array(self):
        key = estimate_key(self.score.note_array)
        self.assertTrue(key == "Am", "Incorrect key")
