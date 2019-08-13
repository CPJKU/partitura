"""

This file contains test functions for the partitura.musicxml module.

"""

import unittest

from . import MUSICXML_PATH, MUSICXML_IMPORT_EXPORT_TESTFILES

from partitura.directions import parse_words
from partitura import load_musicxml, to_musicxml
import partitura.score as score


class TestDirectionParser(unittest.TestCase):
    """
    Test if the direction parser gives the expected results for some directions
    """

    cases = [
        ('Andante', score.ConstantTempoDirection),
        ('ligato', score.ConstantArticulationDirection),
        ('sempre cresc', score.DynamicLoudnessDirection),
        ('poco a poco rallentando', score.DynamicTempoDirection),
        ('this is not a direction', score.Words)
        ]
        
    def test_parser(self):
        for words, cls in self.cases:
            self.assertEqual(type(parse_words(words)), cls)


class TestMusicXML(unittest.TestCase):
    """
    Test if importing and subsequent exporting restores the original input character by character
    """

    def test_import_export(self):
        for fn in MUSICXML_IMPORT_EXPORT_TESTFILES:
            with open(fn) as f:
                parts = load_musicxml(f, validate=False)
                result = to_musicxml(parts).decode('UTF-8')
                f.seek(0)
                target = f.read()
                equal = target == result
                # self.assertEqual(target, result, "Should be equal")
                self.assertTrue(equal, "")


if __name__ == '__main__':
    unittest.main()
