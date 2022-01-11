import unittest

from partitura import EXAMPLE_MUSICXML
from partitura import load_musicxml
from partitura.musicanalysis import estimate_key
from partitura.utils import key_name_to_fifths_mode, fifths_mode_to_key_name


class KeyModeComputation(unittest.TestCase):
    def test_key_name_to_fifths_mode(self):
        self.assertEqual((0, 'minor'), key_name_to_fifths_mode('Am'))
        self.assertEqual((0, 'major'), key_name_to_fifths_mode('C'))
        self.assertEqual((3, 'major'), key_name_to_fifths_mode('A'))
        self.assertEqual((10, "major"), key_name_to_fifths_mode("A#"))
        self.assertEqual((8, "minor"), key_name_to_fifths_mode("E#m"))

    def test_fifths_modes_to_key_name(self):
        self.assertEqual("Am", fifths_mode_to_key_name(0, 'minor'))
        self.assertEqual("C", fifths_mode_to_key_name(0, 'major'))
        self.assertEqual("A", fifths_mode_to_key_name(3, 'major'))
        self.assertEqual("F", fifths_mode_to_key_name(-1, 1))


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




if __name__ == "__main__":
    unittest.main()
