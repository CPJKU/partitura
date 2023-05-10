import unittest
from partitura import EXAMPLE_MUSICXML
from partitura import load_score
from partitura.score import OctaveShiftDirection
import numpy as np
from tests import OCTAVE_SHIFT_TESTFILES


class OctaveShift(unittest.TestCase):
    def test_octave_shift(self):
        part = load_score(EXAMPLE_MUSICXML)[0]
        na = part.note_array(include_pitch_spelling=True, include_staff=True)
        # na["octave"][na["staff"] == 1] += 1
        # Octave shift is applied to the 1st staff
        shift_part = load_score(OCTAVE_SHIFT_TESTFILES[0])[0]
        octave_post_shift = shift_part.note_array(include_pitch_spelling=True)["octave"]
        self.assertEqual(np.all(na["octave"] == octave_post_shift), True)


if __name__ == '__main__':
    unittest.main()
