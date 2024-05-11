from partitura import load_rntxt, load_kern
from partitura.score import RomanNumeral
from partitura import load_musicxml
import urllib.request
import unittest
import os
from tests import KERN_PATH


class TextRNtxtImport(unittest.TestCase):

    def test_chorale_001_from_url(self):
        score_path = os.path.join(KERN_PATH, "chor228.krn")
        rntxt_url = "https://raw.githubusercontent.com/MarkGotham/When-in-Rome/master/Corpus/Early_Choral/Bach%2C_Johann_Sebastian/Chorales/228/analysis.txt"
        score = load_kern(score_path)
        rn_part = load_rntxt(rntxt_url, score, return_part=True)
        romans = list(rn_part.iter_all(RomanNumeral))
        roots = [r.root for r in romans]
        bass = [r.bass_note for r in romans]
        primary_degree = [r.primary_degree for r in romans]
        secondary_degree = [r.secondary_degree for r in romans]
        local_key = [r.local_key for r in romans]
        quality = [r.quality for r in romans]
        inversion = [r.inversion for r in romans]
        expected_roots = ['A', 'A', 'E', 'A', 'A', 'G', 'C', 'C', 'G', 'G', 'A', 'A', 'E', 'E', 'E', 'E', 'A', 'D', 'G#', 'A', 'E', 'A', 'D', 'A', 'B']
        expected_bass = ['A', 'A', 'G#', 'A', 'A', 'B', 'C', 'E', 'G', 'G', 'A', 'A', 'E', 'E', 'E', 'D', 'C', 'C', 'B', 'A', 'E', 'A', 'F', 'E', 'D']
        expected_pdegree = ['i', 'i', 'V', 'i', 'vi', 'V', 'I', 'I', 'V', 'V', 'vi', 'i', 'V', 'V', 'V', 'V', 'i', 'IV', 'viio', 'i', 'V', 'i', 'iv', 'i', 'iio']
        expected_sdegree = ['i', 'i', 'i', 'i', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i']
        expected_lkey = ['a', 'a', 'a', 'a', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a']
        expected_quality = ['min', 'min', 'maj', 'min', 'min', 'maj', 'maj', 'maj', 'maj', '7', 'min', 'min', 'maj', 'maj', 'maj', '7', 'min', '7', 'dim', 'min', 'maj', 'min', 'min', 'min', 'dim7']
        expected_inversion = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 1, 3, 1, 0, 0, 0, 1, 2, 1]
        self.assertEqual(roots, expected_roots)
        self.assertEqual(bass, expected_bass)
        self.assertEqual(primary_degree, expected_pdegree)
        self.assertEqual(secondary_degree, expected_sdegree)
        self.assertEqual(local_key, expected_lkey)
        self.assertEqual(quality, expected_quality)
        self.assertEqual(inversion, expected_inversion)

