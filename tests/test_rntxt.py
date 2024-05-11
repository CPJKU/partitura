from partitura import load_rntxt
from partitura import load_musicxml
import urllib.request
import unittest


class TextRNtxtImport(unittest.TestCase):
    def text_chorale_from_url(self):
        score_url = ""
        rntxt_url = ""
        with urllib.request.urlopen(score_url) as response:
            data = response.read().decode()
        score = load_musicxml(data)
        rn_part = load_rntxt(rntxt_url, score, return_part=True)
        self.assertTrue()

