import unittest
from tests import METRICAL_POSITION_TESTFILES, MUSICXML_IMPORT_EXPORT_TESTFILES, MEI_TESTFILES
from partitura import load_musicxml, load_mei
from partitura.musicanalysis import make_note_feats


class TestingNoteFeatureExtraction(unittest.TestCase):
    def test_metrical_basis(self):
        for fn in METRICAL_POSITION_TESTFILES:
            part = load_musicxml(fn)
            make_note_feats(part, ['metrical_feature'])

    def test_grace_basis(self):
        fn = [f for f in MEI_TESTFILES if f.endswith("test_grace_note.mei")][0]
        part = load_mei(fn)
        make_note_feats(part, ['grace_feature'])

    def test_all_basis(self):
        for fn in MUSICXML_IMPORT_EXPORT_TESTFILES:
            part = load_musicxml(fn)
            make_note_feats(part, "all")


if __name__ == '__main__':
    unittest.main()
