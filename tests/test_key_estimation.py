import unittest

from partitura import EXAMPLE_MUSICXML
from partitura import load_musicxml
from partitura.musicanalysis import estimate_key
from partitura.utils import key_name_to_fifths_mode, fifths_mode_to_key_name


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

    def key_name_to_fifths_mode(self):
        self.assertEqual(key_name_to_fifths_mode('Am'), (0, 'minor'))
        self.assertEqual(key_name_to_fifths_mode('C'), (0, 'major'))
        self.assertEqual(key_name_to_fifths_mode('A'), (3, 'major'))

    def fifths_modes_to_key_name(self):
        self.assertEqual(fifths_mode_to_key_name(0, 'minor'), "Am")
        self.assertEqual(fifths_mode_to_key_name(0, 'major'), 'C')
        self.assertEqual(fifths_mode_to_key_name(3, 'major'), 'A')
        self.assertEqual(fifths_mode_to_key_name(-1, 1), "F")


if __name__ == "__main__":
    unittest.main()
