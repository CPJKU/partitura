#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for MusicXML import and export.
"""

import logging
import unittest
import os
from tempfile import TemporaryDirectory, TemporaryFile
from fractions import Fraction

from tests import (
    MUSICXML_IMPORT_EXPORT_TESTFILES,
    MUSICXML_SCORE_OBJECT_TESTFILES,
    MUSICXML_TUPLET_ATTRIBUTES_TESTFILES,
    MUSICXML_UNFOLD_TESTPAIRS,
    MUSICXML_UNFOLD_COMPLEX,
    MUSICXML_UNFOLD_VOLTA,
    MUSICXML_UNFOLD_DACAPO,
    MUSICXML_CHORD_FEATURES,
    MUSICXML_IGNORE_INVISIBLE_OBJECTS,
    MXL_TESTFILES,
)

from partitura import load_musicxml, save_musicxml
from partitura.io.importmusicxml import validate_musicxml
from partitura.directions import parse_direction
from partitura.utils import show_diff
import partitura.score as score
from lxml import etree

LOGGER = logging.getLogger(__name__)


class TestIncompleteMeasures(unittest.TestCase):
    """
    Measures should keep their duration, even if they are incomplete or empty
    """

    def test_incomplete_measures(self):
        part = score.Part("P1", quarter_duration=2)
        part.add(score.TimeSignature(4, 4), 0)
        for i in range(4):
            part.add(score.Measure(number=i + 1, name=str(i + 1)), 8 * i, 8 * i + 8)
        # the first measure is incomplete, the second one complete, the others empty
        for i, (start, end) in enumerate(((0, 2), (8, 16))):
            note = score.Note("C", 4, id=f"n{i}", voice=1, staff=1)
            part.add(note, start, end)

        xml = save_musicxml(part)
        self.assertEqual(
            [
                [int(e.findtext("duration")) for e in measure_e.findall("forward")]
                for measure_e in etree.fromstring(xml).findall(".//measure")
            ],
            [[6], [], [8], [8]],
        )

        with TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "test.musicxml")
            save_musicxml(part, fn)
            loaded_part = load_musicxml(fn)[0]
        self.assertEqual(
            [(m.name, m.end.t - m.start.t) for m in loaded_part.measures],
            [(m.name, m.end.t - m.start.t) for m in part.measures],
        )


class TestOmittedSections(unittest.TestCase):
    """
    Test the export of scores with omitted sections
    """

    def make_part(self, notes, omitted, time_signatures=(), key_signatures=()):
        """
        Make a part with 10 measures. The measures in `notes` start with a quarter
        note. Signatures are tuples starting with the number of the measure.
        """
        time_signatures = {1: (4, 4), **{m: ts for m, *ts in time_signatures}}
        part = score.Part("P1", quarter_duration=2)
        for staff in (1, 2):
            part.add(score.Clef(staff, "G", 2, 0), 0)

        starts = [0]
        for m in range(1, 11):
            if m in time_signatures:
                beats, beat_type = time_signatures[m]
                part.add(score.TimeSignature(beats, beat_type), starts[-1])
            for ks_measure, fifths in key_signatures:
                if ks_measure == m:
                    part.add(score.KeySignature(fifths, "major"), starts[-1])
            if m in notes:
                note = score.Note("C", 4, id=f"n{m}", voice=1, staff=1)
                part.add(note, starts[-1], starts[-1] + 2)
            starts.append(starts[-1] + beats * 8 // beat_type)
            part.add(score.Measure(number=m, name=str(m)), starts[-2], starts[-1])
        for first, last in omitted:
            part.add(score.OmittedSection(), starts[first - 1], starts[last])

        return part

    def export(self, part, **kwargs):
        """
        Get dictionaries with the labels, multiple-measure rests, durations of
        the whole-measure rests, time and key signatures of each measure
        """
        labels, multiple_rests, rests, times, keys = {}, {}, {}, {}, {}
        measures = etree.fromstring(save_musicxml(part, **kwargs)).findall(".//measure")
        # no measure is left out
        self.assertEqual(len(measures), 10)
        for m, measure_e in enumerate(measures, 1):
            for e in measure_e.findall("direction/direction-type/words"):
                labels[m] = e.text
            for e in measure_e.findall(".//multiple-rest"):
                multiple_rests[m] = int(e.text)
            for e in measure_e.findall("note/rest[@measure='yes']"):
                duration = int(e.getparent().findtext("duration"))
                rests.setdefault(m, []).append(duration)
            for e in measure_e.findall("attributes/time"):
                times[m] = (int(e.findtext("beats")), int(e.findtext("beat-type")))
            for e in measure_e.findall("attributes/key"):
                keys[m] = int(e.findtext("fifths"))

        return labels, multiple_rests, rests, times, keys

    def test_omitted_section(self):
        part = self.make_part([1, 10], [(2, 9)])
        n_objects = len(list(part.iter_all()))
        labels, multiple_rests, rests, times, _ = self.export(part)
        # the number of the measure after the omitted ones is displayed
        self.assertEqual(labels, {2: "omitted", 10: "10"})
        self.assertEqual(multiple_rests, {2: 8})
        self.assertEqual(rests, {m: [8, 8] for m in range(2, 10)})
        self.assertEqual(times, {1: (4, 4)})
        # the part is not changed
        self.assertEqual(len(list(part.iter_all())), n_objects)

        # a single measure
        labels, multiple_rests, rests, _, _ = self.export(
            self.make_part([1, 3, 4, 5, 6, 7, 8, 9, 10], [(2, 2)])
        )
        self.assertEqual((labels, multiple_rests), ({2: "omitted", 3: "3"}, {}))
        self.assertEqual(rests, {2: [8, 8]})

    def test_no_omitted_section(self):
        for part in (
            self.make_part([1, 10], []),
            self.make_part(list(range(1, 11)), [(1, 10)]),
        ):
            self.assertEqual(self.export(part), ({}, {}, {}, {1: (4, 4)}, {}))

    def test_several_omitted_sections(self):
        # measures with notes are not displayed as omitted
        part = self.make_part([1, 5, 10], [(1, 10)])
        labels, multiple_rests, rests, _, _ = self.export(part)
        self.assertEqual(labels, {2: "omitted", 5: "5", 6: "omitted", 10: "10"})
        self.assertEqual(multiple_rests, {2: 3, 6: 4})
        self.assertEqual(sorted(rests), [2, 3, 4, 6, 7, 8, 9])

        # empty measures that are not omitted
        part = self.make_part([1, 10], [(2, 3), (6, 8)])
        labels, multiple_rests, rests, _, _ = self.export(part)
        self.assertEqual(multiple_rests, {2: 2, 6: 3})
        self.assertEqual(sorted(rests), [2, 3, 6, 7, 8])

    def test_signatures(self):
        """
        Signatures are only displayed after the omitted section if they changed
        """
        times = [(4, 2, 4), (5, 4, 4), (7, 3, 4)]
        part = self.make_part([1, 9, 10], [(2, 8)], times, [(1, 0), (3, 2), (6, -1)])
        labels, multiple_rests, rests, times, keys = self.export(part)
        self.assertEqual(labels, {2: "omitted", 9: "9"})
        self.assertEqual(multiple_rests, {2: 7})
        # all measures are displayed in the time signature of the first one
        self.assertEqual(rests, {m: [8, 8] for m in range(2, 9)})
        self.assertEqual(times, {1: (4, 4), 9: (3, 4)})
        self.assertEqual(keys, {1: 0, 9: -1})

        # back to the same signatures, at the end of and after the section
        for measure in (8, 9):
            part = self.make_part(
                [1, 9, 10],
                [(2, 8)],
                [(4, 2, 4), (measure, 4, 4)],
                [(1, 0), (3, 2), (measure, 0)],
            )
            _, _, _, times, keys = self.export(part)
            self.assertEqual((times, keys), ({1: (4, 4)}, {1: 0}))

        # signatures at the start of the section and of the next measure
        part = self.make_part(
            [1, 9, 10], [(2, 8)], [(2, 3, 4), (9, 2, 4)], [(2, 1), (9, 3)]
        )
        _, _, rests, times, keys = self.export(part)
        self.assertEqual(rests, {m: [6, 6] for m in range(2, 9)})
        self.assertEqual(times, {1: (4, 4), 2: (3, 4), 9: (2, 4)})
        self.assertEqual(keys, {2: 1, 9: 3})

    def test_not_merged(self):
        """
        Measures are written as they are in the score
        """
        times = [(4, 2, 4), (7, 3, 4)]
        part = self.make_part([1, 9, 10], [(2, 8)], times)
        exported = self.export(part, merge_omitted_sections=False)
        self.assertEqual(exported, ({}, {}, {}, {1: (4, 4), 4: (2, 4), 7: (3, 4)}, {}))

    def test_order_and_validity(self):
        """
        Some software ignores the label after the multiple-measure rest
        """
        part = self.make_part([1, 10], [(2, 9)], [(2, 3, 4), (5, 4, 4)], [(6, 2)])
        with TemporaryDirectory() as tmpdir:
            for merge in (True, False):
                fn = os.path.join(tmpdir, "test.musicxml")
                save_musicxml(part, fn, merge_omitted_sections=merge)
                self.assertTrue(validate_musicxml(fn))
        measure_e = etree.fromstring(save_musicxml(part)).findall(".//measure")[1]
        self.assertEqual(
            [e.tag for e in measure_e],
            ["attributes", "direction", "attributes", "note", "backup", "note"],
        )
        self.assertIsNotNone(measure_e[0].find("time"))
        self.assertIsNotNone(measure_e[2].find("measure-style/multiple-rest"))

        # signatures, number, and note of the measure after the omitted ones
        measure_e = etree.fromstring(save_musicxml(part)).findall(".//measure")[-1]
        self.assertEqual(
            [e.tag for e in measure_e], ["attributes", "direction", "note", "forward"]
        )


class TestDirectionParser(unittest.TestCase):
    """
    Test if the direction parser gives the expected results for some directions
    """

    cases = [
        ("Andante", [score.ConstantTempoDirection]),
        ("ligato", [score.ConstantArticulationDirection]),
        ("sempre cresc", [score.IncreasingLoudnessDirection]),
        ("poco a poco rallentando", [score.DecreasingTempoDirection]),
    ]

    def test_parser(self):
        for words, target in self.cases:
            result = parse_direction(words)
            self.assertEqual(
                len(result),
                len(target),
                '"{}" not parsed correctly into directions'.format(words),
            )
            for res, trg in zip(result, target):
                self.assertEqual(type(res), trg, "")


class TestMusicXML(unittest.TestCase):
    """
    Test if importing and subsequent exporting restores the original input character by character
    """

    def test_import_export(self):
        for fn in MUSICXML_IMPORT_EXPORT_TESTFILES:
            with open(fn) as f:
                parts = load_musicxml(f, validate=False)
                result = save_musicxml(parts).decode("UTF-8")
                f.seek(0)
                target = f.read()
                equal = target == result
                if not equal:
                    show_diff(result, target)
                msg = "Import and export of MusicXML of file {} does not yield identical result".format(
                    fn
                )
                self.assertTrue(equal, msg)

    def test_unfold_timeline(self):
        for fn, fn_target_1, fn_target_2 in MUSICXML_UNFOLD_TESTPAIRS:
            part = load_musicxml(fn, validate=False)[0]
            part = score.unfold_part_maximal(part, update_ids=False)
            # Load Target
            with open(fn_target_1) as f:
                target = f.read()
            # Transform part to musicxml
            result = save_musicxml(part).decode("UTF-8")

            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = "Unfolding part of MusicXML file {} does not yield expected result".format(
                fn
            )
            self.assertTrue(equal, msg)

            # check unfold with update_id
            part = score.unfold_part_maximal(part, update_ids=True)
            result = save_musicxml(part).decode("UTF-8")
            with open(fn_target_2) as f:
                target = f.read()
            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = "Unfolding part of MusicXML file {} does not yield expected result".format(
                fn
            )
            self.assertTrue(equal, msg)

    def test_unfold_complex(self):
        for fn, fn_target in MUSICXML_UNFOLD_COMPLEX:
            part = load_musicxml(fn, validate=False)[0]
            part = score.unfold_part_maximal(part)
            # Load Target
            with open(fn_target) as f:
                target = f.read()
            # Transform part to musicxml
            result = save_musicxml(part).decode("UTF-8")

            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = "Unfolding complex part of MusicXML file {} does not yield expected result".format(
                fn
            )
            self.assertTrue(equal, msg)

    def test_unfold_volta(self):
        for fn, fn_target in MUSICXML_UNFOLD_VOLTA:
            part = load_musicxml(fn, validate=False)[0]
            part = score.unfold_part_maximal(part)
            # Load Target
            with open(fn_target) as f:
                target = f.read()
            # Transform part to musicxml
            result = save_musicxml(part).decode("UTF-8")

            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = "Unfolding volta part of MusicXML file {} does not yield expected result".format(
                fn
            )
            self.assertTrue(equal, msg)

    def test_unfold_dacapo(self):
        for fn, fn_target in MUSICXML_UNFOLD_DACAPO:
            sc = load_musicxml(fn, validate=False)
            part = score.unfold_part_maximal(sc[0])
            # Load Target
            with open(fn_target) as f:
                target = f.read()
            # Transform part to musicxml
            result = save_musicxml(part).decode("UTF-8")

            equal = target == result
            if not equal:
                show_diff(result, target)
            msg = "Unfolding volta part of MusicXML file {} does not yield expected result".format(
                fn
            )
            self.assertTrue(equal, msg)

    def test_export_import_pprint(self):
        # create a part
        part1 = score.Part("My Part")

        # create contents
        divs = 10
        ts = score.TimeSignature(3, 4)
        page1 = score.Page(1)
        system1 = score.System(1)
        measure1 = score.Measure(number=1, name='1')
        note1 = score.Note(step="A", octave=4, voice=1, staff=1)
        rest1 = score.Rest(voice=1, staff=1)
        note2 = score.Note(step="C", octave=5, alter=-1, voice=2, staff=1)

        # and add the contents to the part:
        part1.set_quarter_duration(0, divs)
        part1.add(ts, 0)
        part1.add(measure1, 0, 30)
        part1.add(page1, 0)
        part1.add(system1, 0)
        part1.add(note1, 0, 15)
        part1.add(rest1, 15, 30)
        part1.add(note2, 0, 30)

        score.set_end_times(part1)

        # pretty print the part
        pstring1 = part1.pretty()

        with TemporaryFile() as f:
            # save part to musicxml
            save_musicxml(part1, f)
            f.flush()
            f.seek(0)
            # load part from musicxml
            part2 = load_musicxml(f)[0]

        # pretty print saved/loaded part:
        pstring2 = part2.pretty()

        # test pretty printed strings for equality
        equal = pstring1 == pstring2
        print('equal:', equal)

        if not equal:
            show_diff(pstring1, pstring2)
        msg = "Exported and imported score does not yield identical pretty printed representations"
        self.assertTrue(equal, msg)

    def test_export_import_tuplet(self):
        part1 = make_part_tuplet()
        self._pretty_export_import_pretty_test(part1)

    def test_export_import_slur(self):
        part1 = make_part_slur()
        self._pretty_export_import_pretty_test(part1)

    def test_stem_direction_import(self):
        # test if stem direction is imported correctly for the first note of test_note_ties.xml
        part = load_musicxml(MUSICXML_IMPORT_EXPORT_TESTFILES[0])[0]
        self.assertEqual(part.notes_tied[0].stem_direction, "up")

    def test_tuplet_attributes(self):
        part = load_musicxml(MUSICXML_TUPLET_ATTRIBUTES_TESTFILES[0])[0]
        tuplets = list(part.iter_all(cls=score.Tuplet))
        # Each tuple consists of:
        # (act_notes, norm_notes, act_type, norm_type, act_dots, norm_dots, dur_mult)
        # fmt: off
        real_values = [
            (3, 2, "eighth", "eighth", 0, 0, Fraction(2, 3)),   # classic 3:2 eighth notes tuplet
            (5, 4, "eighth", "eighth", 0, 0, Fraction(4, 5)),   # 5:4 eighth notes tuplet
            (3, 2, "16th", "16th", 0, 0, Fraction(2, 3)),       # 3:2 16th notes tuplet
            (9, 2, "16th", "quarter", 0, 0, Fraction(8, 9)),    # 9 16th notes against 2 quarter notes
            (2, 3, "eighth", "eighth", 0, 0, Fraction(3, 2)),   # classic 2:3 duolet
            (5, 2, "quarter", "quarter", 0, 1, Fraction(3, 5)), # 5 quarter notes in the time of 2 dotted quarter notes
        ]
        # fmt: on
        for tuplet, (n_act, n_norm, t_act, t_norm, d_act, d_norm, dur_mult) in zip(
            tuplets, real_values
        ):
            self.assertEqual(tuplet.actual_notes, n_act)
            self.assertEqual(tuplet.normal_notes, n_norm)
            self.assertEqual(tuplet.actual_type, t_act)
            self.assertEqual(tuplet.normal_type, t_norm)
            self.assertEqual(tuplet.actual_dots, d_act)
            self.assertEqual(tuplet.normal_dots, d_norm)
            self.assertEqual(tuplet.duration_multiplier, dur_mult)

    def _pretty_export_import_pretty_test(self, part1):
        # pretty print the part
        pstring1 = part1.pretty()
        with TemporaryFile() as f:
            # save part to musicxml
            save_musicxml(part1, f)
            f.flush()
            f.seek(0)
            _tmp = f.read().decode("utf8")
            f.seek(0)
            # load part from musicxml
            part2 = load_musicxml(f)[0]

        # pretty print saved/loaded part:
        pstring2 = part2.pretty()

        # test pretty printed strings for equality
        equal = pstring1 == pstring2

        if not equal:
            print("pretty original:")
            print(pstring1)
            print("pretty reloaded:")
            print(pstring2)
            print("saved xml:")
            print(_tmp)
            print("diff:")
            show_diff(pstring1, pstring2)
        msg = "Exported and imported score does not yield identical pretty printed representations"
        self.assertTrue(equal, msg)
        
    def test_score_attribute(self):
        score = load_musicxml(MUSICXML_SCORE_OBJECT_TESTFILES[0])
        test_work_title = "Test Title"
        test_work_number = "Test Opus 1"

        self.assertTrue(score.work_title == test_work_title)
        self.assertTrue(score.work_number == test_work_number)

    def test_chord_duration(self):
        part = load_musicxml(MUSICXML_CHORD_FEATURES[0]).parts[0]
        score.assign_note_ids(part)
        sna = part.note_array()
        
        self.assertEqual(sna[sna['id'] == 'n1']['duration_beat'], 2)
        self.assertEqual(sna[sna['id'] == 'n2']['duration_beat'], 2)
        self.assertEqual(sna[sna['id'] == 'n1']['duration_quarter'], 2)
        self.assertEqual(sna[sna['id'] == 'n2']['duration_quarter'], 2)
    
    def test_import_ignore_invisible_objects(self):
        score_w_invisible = load_musicxml(MUSICXML_IGNORE_INVISIBLE_OBJECTS[0])[0]
        score_wo_invisible = load_musicxml(MUSICXML_IGNORE_INVISIBLE_OBJECTS[0], ignore_invisible_objects=True)[0]

        note_w_invisible_objs = score_w_invisible.note_array()
        note_wo_invisible_objs = score_wo_invisible.note_array()

        # Convert back from structured array to simple tuples as hash problems with set otherwise
        note_w_invisible_objs = set(
            [(n["pitch"], n["onset_beat"], n["duration_beat"]) for n in note_w_invisible_objs]
        )
        note_wo_invisible_objs = set(
            [(n["pitch"], n["onset_beat"], n["duration_beat"]) for n in note_wo_invisible_objs]
        )
        # Make sure all notes in the filtered score are also in the unfiltered score
        self.assertTrue(note_wo_invisible_objs.issubset(note_w_invisible_objs))

        self.assertTrue(len(note_w_invisible_objs) == 11)
        self.assertTrue(len(note_wo_invisible_objs) == 6)

        self.assertTrue(len(score_w_invisible.rests) == 1)
        self.assertTrue(len(score_wo_invisible.rests) == 0)

        self.assertTrue(len(list(score_w_invisible.iter_all(cls=score.Beam))) == 1)
        self.assertTrue(len(list(score_wo_invisible.iter_all(cls=score.Beam))) == 0)

    def test_import_mxl(self):
        for fn in MXL_TESTFILES:
            scr = load_musicxml(fn, force_note_ids=True)
            self.assertTrue(isinstance(scr, score.Score))
            self.assertTrue(isinstance(scr[0], score.Part))

def make_part_slur():
    # create a part
    part = score.Part("My Part")
    # create contents
    divs = 12
    ts = score.TimeSignature(3, 4)
    page1 = score.Page(1)
    system1 = score.System(1)

    note0 = score.Note(id="n0", step="A", octave=4, voice=1, staff=1)
    note1 = score.Note(id="n1", step="A", octave=4, voice=1, staff=1)
    note2 = score.Note(id="n2", step="A", octave=4, voice=1, staff=1)
    note3 = score.Note(id="n3", step="A", octave=4, voice=1, staff=1)

    note4 = score.Note(id="n4", step="A", octave=3, voice=2, staff=1)
    note5 = score.Note(id="n5", step="A", octave=3, voice=2, staff=1)

    slur1 = score.Slur(start_note=note0, end_note=note5)
    slur2 = score.Slur(start_note=note4, end_note=note3)

    # and add the contents to the part:
    part.set_quarter_duration(0, divs)
    part.add(ts, 0)
    part.add(page1, 0)
    part.add(system1, 0)
    part.add(note0, 0, 12)
    part.add(note1, 12, 24)
    part.add(note2, 24, 36)
    part.add(note3, 36, 48)
    part.add(note4, 0, 6)
    part.add(note5, 6, 33)

    part.add(slur1, slur1.start_note.start.t, slur1.end_note.end.t)
    part.add(slur2, slur2.start_note.start.t, slur2.end_note.end.t)

    score.add_measures(part)
    score.tie_notes(part)
    score.set_end_times(part)
    return part


def make_part_tuplet():
    # create a part
    part = score.Part("My Part")

    # create contents
    divs = 12
    ts = score.TimeSignature(3, 4)
    page1 = score.Page(1)
    system1 = score.System(1)

    note1 = score.Note(id="n0", step="A", octave=4, voice=1, staff=1)
    rest1 = score.Rest(voice=1, staff=1)
    note2 = score.Note(id="n2", step="C", octave=4, voice=1, staff=1)
    rest2 = score.Rest(id="r0", voice=1, staff=1)

    # and add the contents to the part:
    part.set_quarter_duration(0, divs)
    part.add(ts, 0)
    part.add(page1, 0)
    part.add(system1, 0)
    part.add(note1, 0, 8)
    part.add(rest1, 8, 16)
    part.add(note2, 16, 24)
    part.add(rest2, 24, 36)

    score.add_measures(part)
    score.find_tuplets(part)
    score.set_end_times(part)

    return part


if __name__ == "__main__":
    unittest.main()
