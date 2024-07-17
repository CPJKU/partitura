import unittest
from partitura import load_dcml
from tests import TSV_PATH
import os
import pandas as pd


class ImportDCMLAnnotations(unittest.TestCase):
    def test_tsv_import_from_dcml(self):
        note_path = os.path.join(TSV_PATH, "test_notes.tsv")
        measure_path = os.path.join(TSV_PATH, "test_measures.tsv")
        harmony_path = os.path.join(TSV_PATH, "test_harmonies.tsv")
        score = load_dcml(note_path, measure_path, harmony_path)
        note_lines = pd.read_csv(note_path, sep="\t", header=None)
        self.assertEqual(len(score.parts), 1)
        self.assertEqual(len(score[0].notes), len(note_lines)-1, "Number of notes do not match")


if __name__ == '__main__':
    unittest.main()
