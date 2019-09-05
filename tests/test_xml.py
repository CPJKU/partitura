"""

This file contains test functions for the partitura.musicxml module.

"""

import unittest

from . import MUSICXML_PATH, MUSICXML_IMPORT_EXPORT_TESTFILES, MUSICXML_UNFOLD_TESTPAIRS

from partitura.directions import parse_words
from partitura import load_musicxml, save_musicxml
import partitura.score as score


class TestDirectionParser(unittest.TestCase):
    """
    Test if the direction parser gives the expected results for some directions
    """

    cases = [
        ('Andante', [score.ConstantTempoDirection]),
        ('ligato', [score.ConstantArticulationDirection]),
        ('sempre cresc', [score.DynamicLoudnessDirection]),
        ('poco a poco rallentando', [score.DynamicTempoDirection]),
        ('this is not a direction', [score.Words])
        ]
        
    def test_parser(self):
        for words, target in self.cases:
            result = parse_words(words)
            self.assertEqual(len(result), len(target), f'"{words}" not parsed correctly into directions')
            for res, trg in zip(result, target):
                self.assertEqual(type(res), trg, '')



class TestMusicXML(unittest.TestCase):
    """
    Test if importing and subsequent exporting restores the original input character by character
    """

    def test_import_export(self):
        for fn in MUSICXML_IMPORT_EXPORT_TESTFILES:
            with open(fn) as f:
                parts = load_musicxml(f, validate=False)
                result = save_musicxml(parts).decode('UTF-8')
                f.seek(0)
                target = f.read()
                equal = target == result
                if not equal:
                    show_diff(result, target)
                self.assertTrue(equal, f"Import and export of MusicXML of file {fn} does not yield identical result")

    def test_unfold_timeline(self):
        for fn, fn_target in MUSICXML_UNFOLD_TESTPAIRS:
            parts = load_musicxml(fn, validate=False)
            part = next(score.iter_parts(parts))
            part.timeline = part.unfold_timeline_maximal()
            result = save_musicxml(part).decode('UTF-8')
            with open(fn_target) as f:
                target = f.read()
            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = f"Unfolding timeline of MusicXML file {fn} does not yield expected result"
            self.assertTrue(equal, msg)
                

def show_diff(a, b):
    import difflib
    differ = difflib.Differ()
    for l in differ.compare(a.split(), b.split()):
        print(l)
    

if __name__ == '__main__':
    unittest.main()
