import unittest
from tests import METRICAL_POSITION_TESTFILES
from partitura import load_musicxml
from partitura.musicanalysis import make_note_feats

class TestingNoteFeatureExtraction(unittest.TestCase):
    def test_metrical_basis(self):
        for fn in METRICAL_POSITION_TESTFILES:
            part = load_musicxml(fn)
            basis, names = make_note_feats(part, ['metrical_basis'])


if __name__ == '__main__':
    unittest.main()
