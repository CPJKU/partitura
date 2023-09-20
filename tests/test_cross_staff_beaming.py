import unittest
import os
from tests import MUSICXML_PATH
from partitura import load_musicxml
import numpy as np

EXAMPLE_FILE = os.path.join(MUSICXML_PATH, "test_cross_staff_beaming.musicxml")

class CrossStaffBeaming(unittest.TestCase):
    def test_cross_staff_single_part_musicxml(self):
        score = load_musicxml(EXAMPLE_FILE)
        note_array = score.note_array(include_staff=True)
        expected_staff = np.array([1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1])
        cross_staff_mask = (note_array["pitch"] > 52) & (note_array["pitch"] < 72)
        note_array_staff = note_array[cross_staff_mask]["staff"]
        expected_voice = np.ones(len(note_array_staff), dtype=int)
        note_array_voice = note_array[cross_staff_mask]["voice"]
        self.assertTrue(np.all(note_array_staff == expected_staff))
        self.assertTrue(np.all(note_array_voice == expected_voice))

