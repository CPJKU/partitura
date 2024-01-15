import unittest
from partitura import load_tsv
from tests import TSV_PATH
import os


class ImportDCMLAnnotations(unittest.TestCase):
    def test_tsv_import_from_dcml(self):
        note_path = os.path.join(TSV_PATH, "test_notes.tsv")
        measure_path = os.path.join(TSV_PATH, "test_measures.tsv")
        harmony_path = os.path.join(TSV_PATH, "test_harmonies.tsv")
        score = load_tsv(note_path, measure_path, harmony_path)
        self.assertEqual(len(score.parts), 1)


if __name__ == '__main__':
    unittest.main()
