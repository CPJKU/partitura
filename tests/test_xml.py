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
        divs = score.Divisions(10)
        ts = score.TimeSignature(3, 4)
        page1 = score.Page(1)
        system1 = score.System(1)
        measure1 = score.Measure(number=1)
        note1 = score.Note(step='A', alter=None, octave=4,
                           symbolic_duration=dict(type='quarter', dots=1),
                           voice=1, staff=1)
        rest1 = score.Rest(voice=1, staff=1,
                           symbolic_duration=dict(type='quarter', dots=1))
        note2 = score.Note(step='C', alter=-1, octave=5,
                           symbolic_duration=dict(type='half', dots=0),
                           voice=2, staff=1)
        
        # and add the contents to the part:
        part1.add(0, divs)
        part1.add(0, ts)
        part1.add(0, measure1, end=30)
        part1.add(0, page1)
        part1.add(0, system1)
        part1.add(0, note1, end=15)
        part1.add(15, rest1, end=30)
        part1.add(0, note2, end=30)
        
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


def create_part_from_spec(spec):
    """Create a part from a specification of divisions, time signatures
    and notes.

    This is a helper function for the TestBeatMap test case.

    Parameters
    ----------
    spec : dictionary
        Part specification

    Returns
    -------
    Part
        Part instance
    
    """
    
    
    part = score.Part('beatmaptest')

    for t, divs in spec['divs']:
        part.add(t, score.Divisions(divs))

    for t, num, den in spec['ts']:
        part.add(t, score.TimeSignature(num, den))

    divs_map = part.divisions_map

    for t, dur in spec['notes']:

        sd = score.estimate_symbolic_duration(dur, int(divs_map(t)))
        part.add(t, score.Note(step='A', alter=None, octave=4,
                               symbolic_duration=sd), t+dur)

    # not strictly necessary
    score.add_measures(part)

    return part


class TestBeatMap(unittest.TestCase):
    """Test that divisions and time signatures are handled correctly
    when computing beat times and symbolic durations.

    Each of the test cases specifies the contents of a part in terms
    of divisions, time signatures, and notes, and corresponding
    targets in terms of the symbolic durations of each note and their
    onset times in beats.

    For each test case, the part is constructed according to the
    specification, and the computed symbolic durations and onsets are
    compared to the targets. Furthermore the beat map (mapping
    divisions times to beat times) and inverse beat map (mapping beat
    times to divisions times) are jointly tested for their
    invertibility.
    
    """
    test_cases = [
        {'part_spec': {
            'divs': ((0, 10), # at time=0, divs=10
                     (50, 2)), # at time=40, divs=2
            'ts': ((0, 4, 4), # at time=0, ts=4/4
                   (40, 3, 8)), # at time=40, ts=3/8
            'notes': ((0, 10), # at time=0, dur=10
                      (10, 10),
                      (20, 10),
                      (30, 10),
                      (40, 5),
                      (45, 5),
                      (50, 1),
                      (51, 3))},
         'target': {
             'sym_durs': (
                 'quarter',
                 'quarter',
                 'quarter',
                 'quarter',
                 'eighth',
                 'eighth',
                 'eighth',
                 'quarter.'),
             'onset_beats': (0, 1, 2, 3, 4, 5, 6, 7) }},
        {'part_spec': {
            'divs': ((0, 10), # at time=0, divs=10
                     (45, 2), # at time=40, divs=2
                     (46, 10)), # at time=47, divs=10
            'ts': ((0, 4, 4), # at time=0, ts=4/4
                   (40, 3, 8)), # at time=40, ts=3/8
            'notes': ((0, 10), # at time=0, dur=10
                      (10, 10),
                      (20, 10),
                      (30, 10),
                      (40, 5),
                      (45, 1),
                      (46, 5),
                      (51, 15))},
         'target': {
             'sym_durs': (
                 'quarter',
                 'quarter',
                 'quarter',
                 'quarter',
                 'eighth',
                 'eighth',
                 'eighth',
                 'quarter.'),
             'onset_beats': (0, 1, 2, 3, 4, 5, 6, 7) }},
    ]
    def test_beat_map_cases(self):
        for test_case in self.test_cases:
            part = create_part_from_spec(test_case['part_spec'])
            self._test_symbolic_durations(part, test_case['target']['sym_durs'])
            self._test_note_onsets(part, test_case['target']['onset_beats'])
            self._test_beat_map(part)
    
    def _test_symbolic_durations(self, part, target_durations):
        notes = part.timeline.get_all(score.Note)
        if len(notes) != len(target_durations):
            LOGGER.warning('Skipping incorrect test case (input and targets do not match)')
            return
        
        for target_sd, note in zip(target_durations, notes):
            est_sd = score.format_symbolic_duration(note.symbolic_duration)
            msg = '{} != {} ({})'.format(target_sd, est_sd, note.start.t)
            self.assertEqual(target_sd, est_sd, msg)
    
    def _test_note_onsets(self, part, target_onsets):
        notes = part.timeline.get_all(score.Note)
        if len(notes) != len(target_onsets):
            LOGGER.warning('Skipping incorrect test case (input and targets do not match)')
            return
        beat_map = part.beat_map
        for target_onset, note in zip(target_onsets, notes):
            self.assertAlmostEqual(float(target_onset), beat_map(note.start.t))

    def _test_beat_map(self, part):
        beat_map = part.beat_map
        inv_beat_map = part.inv_beat_map
        test_times = range(part.timeline.first_point.t, part.timeline.last_point.t)
        for t in test_times:
            self.assertAlmostEqual(t, inv_beat_map(beat_map(t)))


if __name__ == '__main__':
    unittest.main()
