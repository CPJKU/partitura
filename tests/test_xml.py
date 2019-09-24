"""

This file contains test functions for MusicXML import and export.

"""

import logging
import unittest
from tempfile import TemporaryFile

from . import MUSICXML_PATH, MUSICXML_IMPORT_EXPORT_TESTFILES, MUSICXML_UNFOLD_TESTPAIRS

from partitura import load_musicxml, save_musicxml
from partitura.directions import parse_direction
from partitura.utils import show_diff
import partitura.score as score

LOGGER = logging.getLogger(__name__)

class TestDirectionParser(unittest.TestCase):
    """
    Test if the direction parser gives the expected results for some directions
    """

    cases = [
        ('Andante', [score.ConstantTempoDirection]),
        ('ligato', [score.ConstantArticulationDirection]),
        ('sempre cresc', [score.DynamicLoudnessDirection]),
        ('poco a poco rallentando', [score.DynamicTempoDirection]),
        ]
        
    def test_parser(self):
        for words, target in self.cases:
            result = parse_direction(words)
            self.assertEqual(len(result), len(target), '"{}" not parsed correctly into directions'.format(words))
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
                msg = "Import and export of MusicXML of file {} does not yield identical result".format(fn)
                self.assertTrue(equal, msg)


    def test_unfold_timeline(self):
        for fn, fn_target in MUSICXML_UNFOLD_TESTPAIRS:
            print(fn, fn_target)
            part = load_musicxml(fn, validate=False)
            part.timeline = score.unfold_timeline_maximal(part.timeline)
            result = save_musicxml(part).decode('UTF-8')
            with open(fn_target) as f:
                target = f.read()
            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = "Unfolding timeline of MusicXML file {} does not yield expected result".format(fn)
            self.assertTrue(equal, msg)

                
    def test_export_import_pprint(self):
        # create a part
        part1 = score.Part('My Part')

        # create contents
        divs = 10
        ts = score.TimeSignature(3, 4)
        page1 = score.Page(1)
        system1 = score.System(1)
        measure1 = score.Measure(number=1)
        note1 = score.Note(step='A', alter=None, octave=4, voice=1, staff=1)
        rest1 = score.Rest(voice=1, staff=1)
        note2 = score.Note(step='C', alter=-1, octave=5, voice=2, staff=1)
        
        # and add the contents to the part:
        part1.timeline.set_quarter_duration(0, divs)
        part1.timeline.add(ts, 0)
        part1.timeline.add(measure1, 0, 30)
        part1.timeline.add(page1, 0)
        part1.timeline.add(system1, 0)
        part1.timeline.add(note1, 0, 15)
        part1.timeline.add(rest1, 15, 30)
        part1.timeline.add(note2, 0, 30)
        
        # pretty print the part
        pstring1 = part1.pretty()

        with TemporaryFile() as f:
            # save part to musicxml
            save_musicxml(part1, f)
            f.flush()
            f.seek(0)
            # load part from musicxml
            part2 = load_musicxml(f)

        # pretty print saved/loaded part:
        pstring2 = part2.pretty()

        # test pretty printed strings for equality
        equal = pstring1 == pstring2

        if not equal:
            show_diff(pstring1, pstring2)
        msg = 'Exported and imported score does not yield identical pretty printed representations'
        self.assertTrue(equal, msg)


if __name__ == '__main__':
    unittest.main()
