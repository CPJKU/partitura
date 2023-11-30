#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Matchfile import
"""
import unittest
import numpy as np
import re

from tests import MATCH_IMPORT_EXPORT_TESTFILES, MOZART_VARIATION_FILES

from partitura.io.matchlines_v1 import (
    MatchInfo as MatchInfoV1,
    MatchScoreProp as MatchScorePropV1,
    MatchSection as MatchSectionV1,
    MatchStime as MatchStimeV1,
    MatchPtime as MatchPtimeV1,
    MatchStimePtime as MatchStimePtimeV1,
    MatchSnote as MatchSnoteV1,
    MatchNote as MatchNoteV1,
    MatchSnoteNote as MatchSnoteNoteV1,
    MatchSnoteDeletion as MatchSnoteDeletionV1,
    MatchInsertionNote as MatchInsertionNoteV1,
    MatchSustainPedal as MatchSustainPedalV1,
    MatchSoftPedal as MatchSoftPedalV1,
    MatchOrnamentNote as MatchOrnamentNoteV1,
)

from partitura.io.matchlines_v0 import (
    MatchInfo as MatchInfoV0,
    MatchMeta as MatchMetaV0,
    MatchSnote as MatchSnoteV0,
    MatchNote as MatchNoteV0,
    MatchSnoteNote as MatchSnoteNoteV0,
    MatchSnoteDeletion as MatchSnoteDeletionV0,
    MatchSnoteTrailingScore as MatchSnoteTrailingScoreV0,
    MatchInsertionNote as MatchInsertionNoteV0,
    MatchHammerBounceNote as MatchHammerBounceNoteV0,
    MatchTrailingPlayedNote as MatchTrailingPlayedNoteV0,
    MatchSustainPedal as MatchSustainPedalV0,
    MatchSoftPedal as MatchSoftPedalV0,
    MatchTrillNote as MatchTrillNoteV0,
)

from partitura.io.matchfile_base import MatchError, MatchLine

from partitura.io.matchfile_utils import (
    FractionalSymbolicDuration,
    Version,
    interpret_version,
    MatchTimeSignature,
    MatchKeySignature,
    to_camel_case,
    to_snake_case,
)

from partitura.utils.music import (
    key_name_to_fifths_mode,
    fifths_mode_to_key_name,
)

from partitura import load_score, load_performance
from partitura.io.importmatch import load_match, get_version, load_matchfile

RNG = np.random.RandomState(1984)


class TestLoadMatch(unittest.TestCase):
    def test_load_matchfile(self):
        """
        Test `load_matchfile`
        """
        for fn in MATCH_IMPORT_EXPORT_TESTFILES:

            # read file
            with open(fn) as f:

                file_contents = f.read().splitlines()

            # parse match file
            match = load_matchfile(fn)

            # Assert that all lines in the matchfile where matched
            self.assertTrue(len(match.lines), len(file_contents))

    def test_load_match(self):
        """
        Test `load_match`
        """
        perf_match, alignment, score_match = load_match(
            filename=MOZART_VARIATION_FILES["match"],
            create_score=True,
            first_note_at_zero=True,
        )

        pna_match = perf_match.note_array()
        sna_match = score_match.note_array()

        perf_midi = load_performance(
            filename=MOZART_VARIATION_FILES["midi"],
            first_note_at_zero=True,
        )

        pna_midi = perf_midi.note_array()
        score_musicxml = load_score(
            filename=MOZART_VARIATION_FILES["musicxml"],
        )

        sna_musicxml = score_musicxml.note_array()
        assert np.all(sna_match['voice'] == sna_musicxml['voice'])

        for note in alignment:

            # check score info in match and MusicXML
            if "score_id" in note:

                idx_smatch = np.where(sna_match["id"] == note["score_id"])[0]
                idx_sxml = np.where(sna_musicxml["id"] == note["score_id"])[0]

                self.assertTrue(
                    sna_match[idx_smatch]["pitch"] == sna_musicxml[idx_sxml]["pitch"]
                )

                self.assertTrue(
                    np.isclose(
                        sna_match[idx_smatch]["onset_beat"],
                        sna_match[idx_sxml]["onset_beat"],
                    )
                )

                self.assertTrue(
                    np.isclose(
                        sna_match[idx_smatch]["duration_beat"],
                        sna_match[idx_sxml]["duration_beat"],
                    )
                )

            # check performance info in match and MIDI
            if "performance_id" in note:

                idx_pmatch = np.where(pna_match["id"] == note["performance_id"])[0]
                idx_pmidi = np.where(pna_midi["id"] == note["performance_id"])[0]

                self.assertTrue(
                    pna_match[idx_pmatch]["pitch"] == pna_midi[idx_pmidi]["pitch"]
                )

                self.assertTrue(
                    np.isclose(
                        pna_match[idx_pmatch]["onset_sec"],
                        pna_match[idx_pmidi]["onset_sec"],
                    )
                )

                self.assertTrue(
                    np.isclose(
                        pna_match[idx_pmatch]["duration_sec"],
                        pna_match[idx_pmidi]["duration_sec"],
                    )
                )

    def test_get_version(self):
        """
        Test get_version
        """
        correct_line = "info(matchFileVersion,1.0.0)."
        version = get_version(correct_line)

        self.assertTrue(version == Version(1, 0, 0))

        # Since version 0.1.0 files do not include
        # a version line, parsing a wrong line returns
        # version 0.1.0
        wrong_line = "other_line"
        version = get_version(wrong_line)

        self.assertTrue(version == Version(0, 1, 0))


def basic_line_test(ml: MatchLine, verbose: bool = False) -> None:
    """
    Test that the matchline has the correct number and type
    of the mandatory attributes.

    Parameters
    ----------
    ml : MatchLine
        The MatchLine to be tested
    verbose: bool
        Print whether each of the attributes of the match line have the correct
        data type
    """

    # check that field names and field types have the same number of elements
    assert len(ml.field_names) == len(ml.field_types)

    # assert that field names have the correct type
    assert isinstance(ml.field_names, tuple)
    assert all([isinstance(fn, str) for fn in ml.field_names])

    # assert that field types have the correct type
    assert isinstance(ml.field_types, tuple)
    assert all([isinstance(dt, (type, tuple)) for dt in ml.field_types])

    # assert that the string and matchline methods have the correct type
    assert isinstance(str(ml), str)
    assert isinstance(ml.matchline, str)

    # assert that a new created MatchLine from the same `matchline`
    # will result in the same `matchline`
    new_ml = ml.from_matchline(ml.matchline, version=ml.version)
    assert new_ml.matchline == ml.matchline

    # assert that the data types of the match line are correct
    assert ml.check_types(verbose)

    # assert that the pattern has the correct type
    assert isinstance(ml.pattern, (re.Pattern, tuple))

    if isinstance(ml.pattern, tuple):
        assert all([isinstance(pt, re.Pattern) for pt in ml.pattern])

    # assert that format fun has the correct type and number of elements
    assert isinstance(ml.format_fun, (dict, tuple))

    if isinstance(ml.format_fun, dict):
        assert len(ml.format_fun) == len(ml.field_names)
        assert all([callable(ff) for _, ff in ml.format_fun.items()])
    elif isinstance(ml.format_fun, tuple):
        assert sum([len(ff) for ff in ml.format_fun]) == len(ml.field_names)
        for ff in ml.format_fun:
            assert all([callable(fff) for _, fff in ff.items()])

    # Test that MatchError is raised for an incorrectly formatted line
    try:
        ml.from_matchline("wrong_line", version=ml.version)
        assert False  # pragma: no cover
    except MatchError:
        assert True


class TestMatchLinesV1(unittest.TestCase):
    """
    Test matchlines for version 1.0.0
    """

    def test_info_lines(self):
        """
        Test parsing and generating global info lines.
        """

        # The following lines are correctly specified, and the parser
        # should be able to 1) parse them without errors and 2) reconstruct
        # exactly same line.
        version_line = "info(matchFileVersion,1.0.0)."
        piece_line = "info(piece,Etude Op. 10 No. 3)."
        scoreFileName_line = "info(scoreFileName,Chopin_op10_no3.musicxml)."
        scoreFilePath_line = (
            "info(scoreFilePath,/path/to/dataset/Chopin_op10_no3.musicxml)."
        )
        midiFileName_line = "info(midiFileName,Chopin_op10_no3_p01.mid)."
        midiFilePath_line = (
            "info(midiFilePath,/path/to/dataset/Chopin_op10_no3_p01.mid)."
        )
        audioFileName_line = "info(audioFileName,Chopin_op10_no3_p01.wav)."
        audioFilePath_line = (
            "info(audioFilePath,/path/to/dataset/Chopin_op10_no3_p01.wav)."
        )
        audioFirstNote_line = "info(audioFirstNote,1.2345)."
        audioLastNote_line = "info(audioLastNote,9.8372)."
        composer_line = "info(composer,Frèdéryk Chopin)."
        performer_line = "info(performer,A. Human Pianist)."
        midiClockUnits_line = "info(midiClockUnits,480)."
        # midiClockUnits_line = "info(midiClockUnits,4000)."
        midiClockRate_line = "info(midiClockRate,500000)."
        approximateTempo_line = "info(approximateTempo,98.2902)."
        subtitle_line = "info(subtitle,Subtitle)."

        matchlines = [
            version_line,
            piece_line,
            scoreFileName_line,
            scoreFilePath_line,
            midiFileName_line,
            midiFilePath_line,
            audioFileName_line,
            audioFilePath_line,
            audioFirstNote_line,
            audioLastNote_line,
            composer_line,
            performer_line,
            midiClockUnits_line,
            midiClockRate_line,
            approximateTempo_line,
            subtitle_line,
        ]

        for i, ml in enumerate(matchlines):
            mo = MatchInfoV1.from_matchline(ml)
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            basic_line_test(mo, verbose=i == 0)
            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchInfoV1.from_matchline(ml, version=Version(0, 1, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        # The following lines should result in an error
        try:
            # This line is not defined as an info line and should raise an error
            notSpecified_line = "info(notSpecified,value)."

            mo = MatchInfoV1.from_matchline(notSpecified_line)
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            # assert that the error was raised
            self.assertTrue(True)

        try:
            # wrong value (string instead of integer)
            midiClockUnits_line = "info(midiClockUnits,wrong_value)."

            mo = MatchInfoV1.from_matchline(midiClockUnits_line)
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            # assert that the error was raised
            self.assertTrue(True)

        try:
            mo = MatchInfoV1(
                version=Version(0, 5, 0),
                attribute="scoreFileName",
                value="score_file.musicxml",
                value_type=str,
                format_fun=lambda x: str(x),
            )
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_info_lines_from_info_v0(self):

        info_lines = [
            r"info(scoreFileName,'op10_3_1.scr').",
            r"info(midiFileName,'op10_3_1#18.mid').",
            "info(midiClockUnits,4000).",
            "info(midiClockRate,500000).",
            "info(approximateTempo,34.0).",
            "info(subtitle,[]).",
        ]

        for i, ml in enumerate(info_lines):
            mo_v0 = MatchInfoV0.from_matchline(ml, version=Version(0, 1, 0))

            mo_v1 = MatchInfoV1.from_instance(mo_v0, version=Version(1, 0, 0))
            basic_line_test(mo_v1)

    def test_score_prop_lines(self):

        keysig_line = "scoreprop(keySignature,E,0:2,1/8,-0.5000)."
        keysig_line2 = "scoreprop(keySignature,E/C#,0:2,1/8,-0.5000)."
        timesig_line = "scoreprop(timeSignature,2/4,0:2,1/8,-0.5000)."

        directions_line = "scoreprop(directions,[Allegro],0:2,1/8,-0.5000)."

        beatsubdivision_line = "scoreprop(beatSubDivision,[2],0:2,1/8,-0.5000)."

        matchlines = [
            keysig_line,
            keysig_line2,
            timesig_line,
            directions_line,
            beatsubdivision_line,
        ]

        for ml in matchlines:
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            mo = MatchScorePropV1.from_matchline(ml, version=Version(1, 0, 0))
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

        # not defined attribute
        try:
            wrong_attribute_line = "scoreprop(wrongAttribute,2,0:2,1/8,-0.5000)."
            mo = MatchScorePropV1.from_matchline(
                wrong_attribute_line, version=Version(1, 0, 0)
            )
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchScorePropV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        try:
            mo = MatchScorePropV1(
                version=Version(0, 5, 0),
                attribute="keySignature",
                value="E",
                value_type=str,
                format_fun=lambda x: str(x),
                measure=1,
                beat=0,
                offset=FractionalSymbolicDuration(0),
                time_in_beats=0.0,
            )
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_score_prop_lines_from_info_v0(self):

        info_lines = [
            "info(keySignature,[en,major]).",
            "info(timeSignature,2/4).",
            "info(beatSubdivision,[2,4]).",
        ]

        for i, ml in enumerate(info_lines):
            mo_v0 = MatchInfoV0.from_matchline(ml, version=Version(0, 1, 0))

            mo_v1 = MatchScorePropV1.from_instance(mo_v0, version=Version(1, 0, 0))
            basic_line_test(mo_v1)

            for fn in mo_v0.field_names:
                # Some attributes have different name/spelling in different versions.
                if fn in mo_v1.field_names and fn != "Attribute":
                    self.assertTrue(getattr(mo_v0, fn) == getattr(mo_v1, fn))

    def test_section_lines(self):

        section_lines = [
            "section(0.0000,100.0000,0.0000,100.0000,[end]).",
            "section(100.0000,200.0000,0.0000,100.0000,[fine,volta end]).",
            "section(100.0000,200.0000,0.0000,100.0000,[volta end]).",
            "section(100.0000,200.0000,0.0000,100.0000,[repeat left]).",
        ]

        for ml in section_lines:
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            mo = MatchSectionV1.from_matchline(ml)
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

        # Check version (an error should be raised for old versions)
        try:
            mo = MatchSectionV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover

        except ValueError:
            self.assertTrue(True)

        # Check that incorrectly formatted line results in a match error
        try:
            # Line does not have [] for the end annotations
            wrong_line = "section(0.0000,100.0000,0.0000,100.0000,end)."
            mo = MatchSectionV1.from_matchline(wrong_line)
            self.assertTrue(False)  # pragma: no cover
        except MatchError:
            self.assertTrue(True)

    def test_stime_lines(self):

        stime_lines = [
            "stime(1:1,0,0.0000,[beat])",
            "stime(2:1,0,4.0000,[downbeat,beat])",
            "stime(0:3,0,-1.0000,[beat])",
        ]

        for ml in stime_lines:

            mo = MatchStimeV1.from_matchline(ml, version=Version(1, 0, 0))
            basic_line_test(mo)

            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchStimeV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        # Wrong version
        try:
            mo = MatchStimeV1(
                version=Version(0, 5, 0),
                measure=1,
                beat=1,
                offset=FractionalSymbolicDuration(0),
                onset_in_beats=0.0,
                annotation_type=["beat", "downbeat"],
            )
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_ptime_lines(self):

        stime_lines = [
            "ptime([1000,1001,999]).",
            "ptime([765]).",
            "ptime([3141592]).",
        ]

        for ml in stime_lines:

            mo = MatchPtimeV1.from_matchline(ml, version=Version(1, 0, 0))
            basic_line_test(mo)

            self.assertTrue(mo.matchline == ml)

            self.assertTrue(isinstance(mo.Onset, float))

        # An error is raised if parsing the wrong version
        try:
            mo = MatchPtimeV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        # Wrong version
        try:
            mo = MatchPtimeV1(
                version=Version(0, 5, 0),
                onsets=[8765, 8754],
            )
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_stimeptime_lines(self):

        stime_lines = [
            "stime(1:1,0,0.0000,[beat])-ptime([1000,1001,999]).",
            "stime(2:1,0,4.0000,[downbeat,beat])-ptime([765]).",
            "stime(0:3,0,-1.0000,[beat])-ptime([3141592]).",
        ]

        for ml in stime_lines:

            mo = MatchStimePtimeV1.from_matchline(ml, version=Version(1, 0, 0))
            basic_line_test(mo)

            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchStimePtimeV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_snote_lines(self):
        
        snote_lines = [
            "snote(n1,[B,n],3,0:2,1/8,1/8,-0.5000,0.0000,[v1])",
            "snote(n3,[G,#],3,1:1,0,1/16,0.0000,0.2500,[v3])",
            "snote(n1,[E,n],4,1:1,0,1/4,0.0000,1.0000,[arp])",
            "snote(n143,[B,b],5,7:2,2/16,1/8,25.5000,26.0000,[s,stacc])",
            "snote(n781,[R,-],-,36:3,0,1/4,107.0000,108.0000,[fermata])",
        ]

        output_strings = [
            (
                "MatchSnote\n"
                " Anchor: n1\n"
                " NoteName: B\n"
                " Modifier: 0\n"
                " Octave: 3\n"
                " Measure: 0\n"
                " Beat: 2\n"
                " Offset: 1/8\n"
                " Duration: 1/8\n"
                " OnsetInBeats: -0.5\n"
                " OffsetInBeats: 0.0\n"
                " ScoreAttributesList: ['v1']"
            ),
            (
                "MatchSnote\n"
                " Anchor: n3\n"
                " NoteName: G\n"
                " Modifier: 1\n"
                " Octave: 3\n"
                " Measure: 1\n"
                " Beat: 1\n"
                " Offset: 0\n"
                " Duration: 1/16\n"
                " OnsetInBeats: 0.0\n"
                " OffsetInBeats: 0.25\n"
                " ScoreAttributesList: ['v3']"
            ),
            (
                "MatchSnote\n"
                " Anchor: n1\n"
                " NoteName: E\n"
                " Modifier: 0\n"
                " Octave: 4\n"
                " Measure: 1\n"
                " Beat: 1\n"
                " Offset: 0\n"
                " Duration: 1/4\n"
                " OnsetInBeats: 0.0\n"
                " OffsetInBeats: 1.0\n"
                " ScoreAttributesList: ['arp']"
            ),
            (
                "MatchSnote\n"
                " Anchor: n143\n"
                " NoteName: B\n"
                " Modifier: -1\n"
                " Octave: 5\n"
                " Measure: 7\n"
                " Beat: 2\n"
                " Offset: 2/16\n"
                " Duration: 1/8\n"
                " OnsetInBeats: 25.5\n"
                " OffsetInBeats: 26.0\n"
                " ScoreAttributesList: ['s', 'stacc']"
            ),
            (
                "MatchSnote\n"
                " Anchor: n781\n"
                " NoteName: R\n"
                " Modifier: None\n"
                " Octave: None\n"
                " Measure: 36\n"
                " Beat: 3\n"
                " Offset: 0\n"
                " Duration: 1/4\n"
                " OnsetInBeats: 107.0\n"
                " OffsetInBeats: 108.0\n"
                " ScoreAttributesList: ['fermata']"
            ),
        ]

        for ml, strl in zip(snote_lines, output_strings):
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            mo = MatchSnoteV1.from_matchline(ml, version=Version(1, 0, 0))
            # test __str__ method
            self.assertTrue(
                all(
                    [
                        ll.strip() == sl.strip()
                        for ll, sl in zip(str(mo).splitlines(), strl.splitlines())
                    ]
                )
            )
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

            # These notes were taken from a piece in 2/4
            # so the symbolic durations can be converted to beats
            # by multiplying by 4

            dur_from_symbolic = float(mo.Duration) * 4

            self.assertTrue(np.isclose(dur_from_symbolic, mo.DurationInBeats))

            # Test that the DurationSymbolic string produces the same duration
            self.assertTrue(
                FractionalSymbolicDuration.from_string(mo.DurationSymbolic)
                == mo.Duration
            )

            self.assertTrue(isinstance(mo.MidiPitch, (int, type(None))))
            self.assertTrue(
                mo.MidiPitch < 128 if isinstance(mo.MidiPitch, int) else True
            )

        # An error is raised if parsing the wrong version
        try:
            mo = MatchSnoteV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        # Wrong version
        try:
            mo = MatchSnoteV1(
                version=Version(0, 5, 0),
                anchor="n0",
                note_name="C",
                modifier="n",
                octave=4,
                measure=1,
                beat=0,
                offset=FractionalSymbolicDuration(0),
                duration=FractionalSymbolicDuration(1),
                onset_in_beats=0,
                offset_in_beats=1,
                score_attributes_list=[],
            )
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_snote_from_v0(self):

        snote_lines = [
            "snote(n1,[c,n],6,0:3,0/1,1/8,-4.00000,-3.00000,[1])",
            "snote(n726,[f,n],3,45:1,0/1,0/8,264.00000,264.00000,[5,arp])",
            "snote(n714,[a,n],5,44:6,0/1,1/8,263.00000,264.00000,[1])",
            "snote(n1,[b,n],4,0:2,1/8,1/8,-0.50000,0.00000,[1])",
            "snote(n445,[e,n],4,20:2,1/16,1/16,39.25000,39.50000,[4])",
        ]

        for ml in snote_lines:

            for minor_version in (1, 2):
                # assert that the information from the matchline
                # is parsed correctly and results in an identical line
                # to the input match line
                mo_v0 = MatchSnoteV0.from_matchline(
                    ml,
                    version=Version(0, minor_version, 0),
                )

                mo_v1 = MatchSnoteV1.from_instance(mo_v0, version=Version(1, 0, 0))
                basic_line_test(mo_v1)
                self.assertTrue(isinstance(mo_v1, MatchSnoteV1))
                self.assertTrue(mo_v0.version != mo_v1.version)

                for fn in mo_v1.field_names:

                    if fn in mo_v0.field_names:
                        self.assertTrue(getattr(mo_v0, fn) == getattr(mo_v1, fn))

        try:
            mo_v1 = MatchSnoteV1.from_instance(mo_v0, version=Version(0, 5, 0))

            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        try:
            wrong_mo = MatchSnoteNoteV1.from_matchline(
                "snote(n1,[B,n],3,0:2,1/8,1/8,-0.5000,0.0000,[v1])-note(0,47,39940,42140,44,0,0)."
            )
            mo_v1 = MatchSnoteV1.from_instance(wrong_mo)

            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_note_lines(self):

        note_lines = [
            "note(0,47,46710,58040,26,0,0).",
            "note(13,51,72850,88210,45,0,0).",
            "note(32,28,103220,114320,37,0,0).",
            "note(65,51,153250,157060,60,0,0).",
        ]

        for ml in note_lines:
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            mo = MatchNoteV1.from_matchline(ml, version=Version(1, 0, 0))
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

            # assert that the data types of the match line are correct
            self.assertTrue(mo.check_types())

        # An error is raised if parsing the wrong version
        try:
            mo = MatchNoteV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_note_from_v0(self):

        # Lines taken from original version of
        # Chopin Op. 38 in old Vienna4x22
        note_lines = [
            "note(1,[c,n],6,39060.00,39890.00,38).",
            "note(6,[c,n],5,48840.00,49870.00,26).",
            "note(17,[c,n],5,72600.00,75380.00,26).",
            "note(32,[b,b],5,93030.00,95050.00,32).",
            "note(85,[b,b],3,162600.00,164950.00,27).",
            "note(132,[c,n],5,226690.00,227220.00,34).",
            "note(179,[b,b],4,280360.00,282310.00,35).",
        ]

        for ml in note_lines:
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line

            for minor_version in range(1, 3):
                mo_v0 = MatchNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )

                mo_v1 = MatchNoteV1.from_instance(mo_v0, version=Version(1, 0, 0))
                basic_line_test(mo_v1)
                self.assertTrue(isinstance(mo_v1, MatchNoteV1))
                self.assertTrue(mo_v0.version != mo_v1.version)

                for fn in mo_v1.field_names:

                    if fn in mo_v0.field_names:
                        self.assertTrue(getattr(mo_v0, fn) == getattr(mo_v1, fn))

        try:
            mo_v1 = MatchNoteV1.from_instance(mo_v0, version=Version(0, 5, 0))

            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        try:
            wrong_mo = MatchSnoteNoteV1.from_matchline(
                "snote(n1,[B,n],3,0:2,1/8,1/8,-0.5000,0.0000,[v1])-note(0,47,39940,42140,44,0,0)."
            )
            mo_v1 = MatchNoteV1.from_instance(wrong_mo)

            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_snotenote_lines(self):

        snotenote_lines = [
            "snote(n1,[B,n],3,0:2,1/8,1/8,-0.5000,0.0000,[v1])-note(0,47,39940,42140,44,0,0).",
            "snote(n443,[B,n],2,20:2,0,1/16,39.0000,39.2500,[v7])-note(439,35,669610,679190,28,0,0).",
            "snote(n444,[B,n],3,20:2,1/16,1/16,39.2500,39.5000,[v3])-note(441,47,673620,678870,27,0,0).",
            "snote(n445,[E,n],3,20:2,1/16,1/16,39.2500,39.5000,[v4])-note(442,40,673980,678130,19,0,0).",
            "snote(n446,[G,#],3,20:2,1/8,1/16,39.5000,39.7500,[v3])-note(443,44,678140,683800,23,0,0).",
            "snote(n447,[E,n],2,20:2,1/8,3/8,39.5000,41.0000,[v7])-note(444,28,678170,704670,22,0,0).",
            "snote(n448,[B,n],3,20:2,3/16,1/16,39.7500,40.0000,[v3])-note(445,47,683550,685070,30,0,0).",
            "snote(n449,[B,n],2,20:2,3/16,5/16,39.7500,41.0000,[v6])-note(446,35,683590,705800,18,0,0).",
            "snote(n450,[G,#],4,21:1,0,0,40.0000,40.0000,[v1,grace])-note(447,56,691330,694180,38,0,0).",
            "snote(n451,[F,#],4,21:1,0,0,40.0000,40.0000,[v1,grace])-note(450,54,693140,695700,44,0,0).",
            "snote(n452,[E,n],4,21:1,0,1/4,40.0000,41.0000,[v1])-note(451,52,695050,705530,40,0,0).",
            "snote(n453,[G,#],3,21:1,0,1/4,40.0000,41.0000,[v3])-note(449,44,691800,703570,28,0,0).",
        ]

        for i, ml in enumerate(snotenote_lines):

            mo = MatchSnoteNoteV1.from_matchline(ml, version=Version(1, 0, 0))
            basic_line_test(mo, verbose=i == 0)
            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchSnoteNoteV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_deletion_lines(self):

        deletion_lines = [
            "snote(n158,[A,n],3,8:2,0,1/16,15.0000,15.2500,[v3])-deletion.",
            "snote(n270,[F,#],4,14:1,0,1/16,26.0000,26.2500,[v2])-deletion.",
            "snote(n323,[A,#],3,16:1,0,1/16,30.0000,30.2500,[v4])-deletion.",
            "snote(n325,[E,n],3,16:1,0,1/16,30.0000,30.2500,[v6])-deletion.",
            "snote(n328,[A,#],4,16:1,1/16,1/16,30.2500,30.5000,[v2])-deletion.",
            "snote(n331,[F,#],3,16:1,1/16,1/16,30.2500,30.5000,[v5])-deletion.",
            "snote(n99-1,[E,n],4,8:3,0,1/8,44.0000,45.0000,[staff1])-deletion.",
            "snote(n99-2,[E,n],4,16:3,0,1/8,92.0000,93.0000,[staff1])-deletion.",
            "snote(n238-1,[A,n],4,26:4,0,1/4,153.0000,155.0000,[staff1])-deletion.",
            "snote(n238-2,[A,n],4,36:4,0,1/4,213.0000,215.0000,[staff1])-deletion.",
        ]

        for ml in deletion_lines:

            mo = MatchSnoteDeletionV1.from_matchline(ml, version=Version(1, 0, 0))

            basic_line_test(mo)

            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchSnoteDeletionV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_insertion_lines(self):

        insertion_lines = [
            "insertion-note(50,40,109820,118820,1,0,0).",
            "insertion-note(51,57,109940,113160,1,0,0).",
            "insertion-note(82,47,164500,164830,1,0,0).",
            "insertion-note(125,44,240630,243190,1,0,0).",
            "insertion-note(172,53,230380,230950,1,0,0).",
            "insertion-note(263,26,322180,322800,81,0,0).",
            "insertion-note(292,61,344730,347960,116,0,0).",
            "insertion-note(241,56,328460,333340,17,0,0).",
            "insertion-note(101,56,210170,211690,1,0,0).",
            "insertion-note(231,45,482420,485320,1,0,0).",
            "insertion-note(307,56,636010,637570,1,0,0).",
            "insertion-note(340,56,693470,696110,1,0,0).",
            "insertion-note(445,58,914370,917360,1,0,0).",
            "insertion-note(193,56,235830,236270,98,0,0).",
            "insertion-note(50,40,143130,156020,1,0,0).",
            "insertion-note(156,40,424930,437570,1,0,0).",
        ]

        for i, ml in enumerate(insertion_lines):

            mo = MatchInsertionNoteV1.from_matchline(ml, version=Version(1, 0, 0))
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchInsertionNoteV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_ornament_lines(self):

        ornament_lines = [
            "ornament(1156-1,[trill])-note(566,78,41161,41246,71,0,0).",
            "ornament(1158-1,[trill])-note(573,78,41447,41558,73,0,0).",
            "ornament(1158-1,[trill])-note(574,77,41536,41622,63,0,0).",
            "ornament(1252-1,[trill])-note(664,77,47659,47798,56,0,0).",
        ]

        for ml in ornament_lines:

            mo = MatchOrnamentNoteV1.from_matchline(ml, version=Version(1, 0, 0))
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchOrnamentNoteV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_sustain_lines(self):

        sustain_lines = [
            "sustain(711360,22).",
            "sustain(711440,21).",
            "sustain(711520,19).",
            "sustain(711600,18).",
            "sustain(711680,17).",
            "sustain(711760,16).",
            "sustain(711840,16).",
            "sustain(712080,15).",
            "sustain(712560,15).",
            "sustain(715280,14).",
            "sustain(717920,14).",
            "sustain(720080,13).",
            "sustain(721760,13).",
            "sustain(731920,13).",
        ]

        for i, ml in enumerate(sustain_lines):

            mo = MatchSustainPedalV1.from_matchline(ml, version=Version(1, 0, 0))
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchSustainPedalV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_soft_lines(self):

        soft_lines = [
            "soft(2,4).",
            "soft(180,4).",
            "soft(182,3).",
            "soft(184,4).",
            "soft(282,4).",
            "soft(284,3).",
            "soft(286,4).",
            "soft(812,4).",
            "soft(814,3).",
            "soft(816,4).",
            "soft(984,4).",
            "soft(986,3).",
            "soft(988,4).",
            "soft(1006,4).",
            "soft(1008,3).",
        ]

        for i, ml in enumerate(soft_lines):

            mo = MatchSoftPedalV1.from_matchline(ml, version=Version(1, 0, 0))
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchSoftPedalV1.from_matchline(ml, version=Version(0, 5, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)


class TestMatchLinesV0(unittest.TestCase):
    def test_info_lines_v_0_1_0(self):
        matchlines = [
            r"info(scoreFileName,'op10_3_1.scr').",
            r"info(midiFileName,'op10_3_1#18.mid').",
            "info(midiClockUnits,4000).",
            "info(midiClockRate,500000).",
            "info(keySignature,[en,major]).",
            "info(timeSignature,2/4).",
            "info(beatSubdivision,[2,4]).",
            "info(tempoIndication,[lento,ma,non,troppo]).",
            "info(approximateTempo,34.0).",
            "info(subtitle,[]).",
        ]

        for i, ml in enumerate(matchlines):

            for minor in (1, 2):
                mo = MatchInfoV0.from_matchline(ml, version=Version(0, minor, 0))
                # assert that the information from the matchline
                # is parsed correctly and results in an identical line
                # to the input match line
                basic_line_test(mo, verbose=(i == 0) and (minor == 1))
            self.assertTrue(mo.matchline == ml)

        # The following lines should result in an error
        try:
            # This line is not defined as an info line and should raise an error
            notSpecified_line = "info(notSpecified,value)."

            mo = MatchInfoV0.from_matchline(notSpecified_line)
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            # assert that the error was raised
            self.assertTrue(True)

        try:
            # wrong value (string instead of integer)
            midiClockUnits_line = "info(midiClockUnits,wrong_value)."

            mo = MatchInfoV0.from_matchline(midiClockUnits_line)
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            # assert that the error was raised
            self.assertTrue(True)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchInfoV0.from_matchline(ml, version=Version(1, 0, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        try:
            # This is not a valid line and should result in a MatchError
            wrong_line = "wrong_line"
            mo = MatchInfoV0.from_matchline(wrong_line)
            self.assertTrue(False)  # pragma: no cover
        except MatchError:
            self.assertTrue(True)

        try:
            mo = MatchInfoV0(
                version=Version(1, 0, 0),
                attribute="scoreFileName",
                value="'score_file.musicxml'",
                value_type=str,
                format_fun=lambda x: str(x),
            )
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_info_lines_v_0_3_0(self):
        matchlines = [
            r"info(scoreFileName,'op10_3_1.scr').",
            r"info(midiFileName,'op10_3_1#18.mid').",
            "info(midiClockUnits,4000).",
            "info(midiClockRate,500000).",
            "info(keySignature,[C Maj]).",
            "info(timeSignature,2/4).",
            "info(beatSubdivision,[2,4]).",
            "info(tempoIndication,[lento,ma,non,troppo]).",
            "info(approximateTempo,34.0).",
            "info(subtitle,[]).",
            # "info(timeSignature,[4/4,2/2,3/4,2/2,3/4,4/4]).",
            # "info(timeSignature,3/8).",
            # "info(timeSignature,[3/4]).",
        ]

        for i, ml in enumerate(matchlines):
            mo = MatchInfoV0.from_matchline(ml, version=Version(0, 3, 0))
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

    def test_info_lines_v_0_4_0(self):
        matchlines = [
            r"info(scoreFileName,'op10_3_1.scr').",
            r"info(midiFileName,'op10_3_1#18.mid').",
            "info(midiClockUnits,4000).",
            "info(midiClockRate,500000).",
            "info(keySignature,[D Maj/B min]).",
            "info(beatSubdivision,[2,4]).",
            "info(tempoIndication,[lento,ma,non,troppo]).",
            "info(approximateTempo,34.0).",
            "info(subtitle,[]).",
            "info(timeSignature,[4/4,2/2,3/4,2/2,3/4,4/4]).",
            "info(timeSignature,[3/8]).",
            "info(timeSignature,[3/4]).",
        ]

        for i, ml in enumerate(matchlines):

            for minor in (4, 5):
                mo = MatchInfoV0.from_matchline(ml, version=Version(0, minor, 0))
                # assert that the information from the matchline
                # is parsed correctly and results in an identical line
                # to the input match line
                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)

    def test_meta_lines(self):

        meta_lines = [
            "meta(timeSignature,3/8,1,-1.5).",
            "meta(keySignature,F Maj/D min,1,-1.5).",
            "meta(timeSignature,2/2,1,-1.0).",
            "meta(keySignature,E Maj/C# min,1,-1.0).",
            "meta(keySignature,G Maj/E min,49,92.0).",
            "meta(keySignature,E Maj/C# min,85,164.0).",
        ]

        for ml in meta_lines:
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            mo = MatchMetaV0.from_matchline(ml, version=Version(0, 3, 0))
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

        # not defined attribute
        try:
            wrong_attribute_line = "meta(wrongAttribute,value,1,-1.5)."
            mo = MatchMetaV0.from_matchline(
                wrong_attribute_line, version=Version(0, 3, 0)
            )
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchMetaV0.from_matchline(ml, version=Version(1, 0, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        try:
            mo = MatchMetaV0(
                version=Version(1, 0, 0),
                attribute="keySignature",
                value="E",
                value_type=str,
                format_fun=lambda x: str(x),
                measure=1,
                time_in_beats=0.0,
            )
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_snote_lines_v0_1_0(self):

        snote_lines = [
            "snote(n1,[c,n],6,0:3,0/1,1/8,-4.00000,-3.00000,[1])",
            "snote(n726,[f,n],3,45:1,0/1,0/8,264.00000,264.00000,[5,arp])",
            "snote(n714,[a,n],5,44:6,0/1,1/8,263.00000,264.00000,[1])",
            "snote(n1,[b,n],4,0:2,1/8,1/8,-0.50000,0.00000,[1])",
            "snote(n445,[e,n],4,20:2,1/16,1/16,39.25000,39.50000,[4])",
        ]

        for ml in snote_lines:

            for minor_version in (1, 2):
                # assert that the information from the matchline
                # is parsed correctly and results in an identical line
                # to the input match line
                mo = MatchSnoteV0.from_matchline(
                    ml,
                    version=Version(0, minor_version, 0),
                )
                basic_line_test(mo)
                # print(mo.matchline, ml)
                self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchSnoteV0.from_matchline(ml, version=Version(1, 0, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        try:
            # This is not a valid line and should result in a MatchError
            wrong_line = "wrong_line"
            mo = MatchSnoteV0.from_matchline(wrong_line, version=Version(0, 1, 0))
            self.assertTrue(False)  # pragma: no cover
        except MatchError:
            self.assertTrue(True)

        # Wrong version
        try:
            mo = MatchSnoteV0(
                version=Version(1, 0, 0),
                anchor="n0",
                note_name="c",
                modifier="n",
                octave=4,
                measure=1,
                beat=0,
                offset=FractionalSymbolicDuration(0),
                duration=FractionalSymbolicDuration(1),
                onset_in_beats=0,
                offset_in_beats=1,
                score_attributes_list=[],
            )
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_snote_lines_v0_3_0(self):

        snote_lines = [
            "snote(n1,[e,n],4,1:1,0,1/4,0.0,1.0,[arp])",
            "snote(n16,[e,n],5,1:4,1/16,1/16,3.25,3.5,[s])",
            "snote(n29,[c,n],5,2:3,0,1/16,6.0,6.25,[s,trill])",
            "snote(n155,[a,n],5,8:1,0,0,28.0,28.0,[s,grace])",
            "snote(n187,[g,n],5,9:2,3/16,1/16,33.75,34.0,[s,stacc])",
            # example of rest included in the original Batik dataset
            "snote(n84,[r,-],-,8:6,0,1/8,47.0,48.0,[fermata])",
        ]

        for ml in snote_lines:

            for minor_version in (3,):
                # assert that the information from the matchline
                # is parsed correctly and results in an identical line
                # to the input match line
                mo = MatchSnoteV0.from_matchline(
                    ml,
                    version=Version(0, minor_version, 0),
                )
                basic_line_test(mo)

                self.assertTrue(mo.matchline == ml)

    def test_snote_lines_v0_5_0(self):

        snote_lines = [
            "snote(n211-1,[E,b],5,20:2,0,1/8,58.0,58.5,[staff1,trill])",
            "snote(n218-1,[A,b],2,20:3,0,1/4,59.0,60.0,[staff2])",
            "snote(n224-2,[A,b],5,36:3,0,1/8,107.0,107.5,[staff1])",
            "snote(n256-2,[E,b],4,38:3,0,1/4,113.0,114.0,[staff2])",
        ]

        for ml in snote_lines:

            for minor_version in (4, 5):
                # assert that the information from the matchline
                # is parsed correctly and results in an identical line
                # to the input match line
                mo = MatchSnoteV0.from_matchline(
                    ml,
                    version=Version(0, minor_version, 0),
                )
                basic_line_test(mo)
                # print(mo.matchline, ml)
                self.assertTrue(mo.matchline == ml)

    def test_note_lines_v_0_4_0(self):

        note_lines = [
            "note(0,[A,n],2,500,684,684,41).",
            "note(1,[C,#],3,704,798,798,48).",
            "note(11,[E,n],6,1543,1562,1562,72).",
            "note(25,[E,n],4,3763,4020,4020,40).",
            "note(102,[F,#],4,13812,14740,14740,61).",
            "note(194,[D,n],5,27214,27272,27272,69).",
            "note(n207,[F,n],5,20557,20635,20682,72).",
            "note(n214,[G,n],5,21296,21543,21543,75).",
        ]

        for ml in note_lines:
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line

            for minor_version in (4, 5):
                mo = MatchNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )
                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)
                # check duration and adjusted duration
                self.assertTrue(mo.AdjDuration >= mo.Duration)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchNoteV0.from_matchline(ml, version=Version(1, 0, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        # Wrong version
        try:
            mo = MatchNoteV0(
                version=Version(1, 0, 0),
                id="n0",
                note_name="C",
                modifier=0,
                octave=4,
                onset=0,
                offset=400,
                velocity=90,
            )
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        try:
            # This is not a valid line and should result in a MatchError
            wrong_line = "wrong_line"
            mo = MatchNoteV0.from_matchline(wrong_line, version=Version(0, 1, 0))
            self.assertTrue(False)  # pragma: no cover
        except MatchError:
            self.assertTrue(True)

    def test_note_lines_v_0_3_0(self):

        note_lines = [
            "note(14,[d,n],5,2239,2355,2355,76).",
            "note(16,[e,n],5,2457,2564,2564,79).",
            "note(29,[c,n],5,3871,3908,3908,62).",
            "note(71,[c,n],5,7942,7983,7983,66).",
            "note(98,[c,#],5,9927,10298,10352,78).",
            "note(964,[a,#],5,91792,91835,91835,69).",
        ]

        for ml in note_lines:
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line

            for minor_version in (3,):
                mo = MatchNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )
                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)
                # check duration and adjusted duration
                self.assertTrue(mo.AdjDuration >= mo.Duration)

    def test_note_lines_v_0_1_0(self):

        # Lines taken from original version of
        # Chopin Op. 38 in old Vienna4x22
        note_lines = [
            "note(1,[c,n],6,39060.00,39890.00,38).",
            "note(6,[c,n],5,48840.00,49870.00,26).",
            "note(17,[c,n],5,72600.00,75380.00,26).",
            "note(32,[b,b],5,93030.00,95050.00,32).",
            "note(85,[b,b],3,162600.00,164950.00,27).",
            "note(132,[c,n],5,226690.00,227220.00,34).",
            "note(179,[b,b],4,280360.00,282310.00,35).",
        ]

        for ml in note_lines:
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line

            for minor_version in range(1, 3):
                mo = MatchNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )
                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)

    def test_snotenote_lines_v_0_1_0(self):

        snotenote_lines = [
            "snote(n1,[c,n],6,0:3,0/1,1/8,-4.00000,-3.00000,[1])-note(1,[c,n],6,39060.00,39890.00,38).",
            "snote(n2,[c,n],5,0:3,0/1,1/8,-4.00000,-3.00000,[4])-note(2,[c,n],5,39120.00,40240.00,34).",
            "snote(n3,[c,n],6,0:4,0/1,2/8,-3.00000,-1.00000,[1])-note(3,[c,n],6,42580.00,44410.00,37).",
            "snote(n4,[c,n],5,0:4,0/1,2/8,-3.00000,-1.00000,[4])-note(4,[c,n],5,42700.00,44250.00,31).",
            "snote(n661,[b,b],4,41:4,0/1,2/8,243.00000,245.00000,[3])-note(661,[b,b],4,943540.00,945410.00,19).",
            "snote(n662,[c,n],4,41:4,0/1,3/8,243.00000,246.00000,[4])-note(662,[c,n],4,943630.00,945900.00,26).",
            "snote(n663,[c,n],3,41:4,0/1,2/8,243.00000,245.00000,[5])-note(663,[c,n],3,943380.00,951590.00,28).",
            "snote(n664,[e,n],5,41:6,0/1,1/8,245.00000,246.00000,[2])-note(664,[e,n],5,950010.00,950840.00,33).",
            "snote(n665,[b,b],4,41:6,0/1,1/8,245.00000,246.00000,[3])-note(665,[b,b],4,950130.00,951570.00,28).",
        ]

        for i, ml in enumerate(snotenote_lines):
            for minor_version in (1, 2):
                mo = MatchSnoteNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )
                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchSnoteNoteV0.from_matchline(ml, version=Version(1, 0, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_snotenote_lines_v_0_3_0(self):

        snotenote_lines = [
            "snote(n1,[e,n],4,1:1,0,1/4,0.0,1.0,[arp])-note(1,[e,n],4,761,1351,1351,60).",
            "snote(n2,[g,n],4,1:1,0,1/4,0.0,1.0,[arp])-note(2,[g,n],4,814,1332,1332,74).",
            "snote(n3,[c,n],3,1:1,0,1/16,0.0,0.25,[])-note(3,[c,n],3,885,943,943,65).",
            "snote(n4,[c,n],5,1:1,0,1/4,0.0,1.0,[s,arp])-note(4,[c,n],5,886,1358,1358,85).",
            "snote(n5,[c,n],4,1:1,1/16,1/16,0.25,0.5,[])-note(5,[c,n],4,1028,1182,1182,67).",
            "snote(n6,[b,n],3,1:1,2/16,1/16,0.5,0.75,[])-note(6,[b,n],3,1151,1199,1199,63).",
            "snote(n7,[c,n],4,1:1,3/16,1/16,0.75,1.0,[])-note(7,[c,n],4,1276,1325,1325,56).",
            "snote(n8,[c,n],3,1:2,0,1/8,1.0,1.5,[])-note(8,[c,n],3,1400,1611,1700,62).",
        ]

        for i, ml in enumerate(snotenote_lines):

            for minor_version in (3,):
                mo = MatchSnoteNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )
                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)

    def test_snote_lines_v_0_4_0(self):

        snotenote_lines = [
            "snote(n1-1,[A,b],4,1:1,0,1/4,0.0,1.0,[staff1,accent])-note(0,[G,#],4,388411,388465,388465,65).",
            "snote(n2-1,[G,n],4,1:2,0,1/8,1.0,1.5,[staff1])-note(1,[G,n],4,389336,389595,389595,35).",
            "snote(n3-1,[A,b],4,1:2,1/8,1/8,1.5,2.0,[staff1])-note(2,[G,#],4,389628,389822,389822,34).",
            "snote(n4-1,[C,n],5,1:3,0,1/8,2.0,2.5,[staff1])-note(3,[C,n],5,389804,389911,389911,44).",
            "snote(n5-1,[B,b],4,1:3,1/8,1/8,2.5,3.0,[staff1])-note(4,[A,#],4,389932,389999,389999,50).",
            "snote(n7-1,[G,n],4,2:1,0,1/8,3.0,3.5,[staff1])-note(5,[G,n],4,390054,390109,390109,49).",
            "snote(n8-1,[A,b],4,2:1,1/8,1/8,3.5,4.0,[staff1])-note(6,[G,#],4,390168,390222,390222,46).",
        ]

        for i, ml in enumerate(snotenote_lines):

            for minor_version in (4, 5):
                mo = MatchSnoteNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )
                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)

    def test_deletion_lines_v_0_4_0(self):

        deletion_lines = [
            "snote(61-2,[E,n],4,13:3,0,1/8,74.0,75.0,[staff2])-deletion.",
            "snote(99-2,[E,n],4,16:3,0,1/8,92.0,93.0,[staff1])-deletion.",
            "snote(167-1,[E,n],4,21:3,0,1/8,122.0,123.0,[staff2])-deletion.",
            "snote(244-1,[E,n],3,26:3,0,1/8,152.0,153.0,[staff2])-deletion.",
            "snote(238-1,[A,n],4,26:4,0,2/8,153.0,155.0,[staff1])-deletion.",
            "snote(167-2,[E,n],4,31:3,0,1/8,182.0,183.0,[staff2])-deletion.",
            "snote(244-2,[E,n],3,36:3,0,1/8,212.0,213.0,[staff2])-deletion.",
            "snote(238-2,[A,n],4,36:4,0,2/8,213.0,215.0,[staff1])-deletion.",
        ]

        for ml in deletion_lines:

            for minor_version in (4, 5):
                mo = MatchSnoteDeletionV0.from_matchline(
                    ml,
                    version=Version(0, minor_version, 0),
                )
                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchSnoteDeletionV0.from_matchline(ml, version=Version(1, 0, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_trailing_score_lines(self):

        # These are all trailing_score_note lines in the Batik dataset
        trailing_score_lines = [
            "snote(n5283,[e,n],4,216:5,1/16/3,1/16/3,1294.1666666666667,1294.3333333333333,[s,stacc])-trailing_score_note.",
            "snote(n5284,[f,#],4,216:5,2/16/3,1/16/3,1294.3333333333333,1294.5,[s,stacc])-trailing_score_note.",
            "snote(n5285,[g,n],4,216:5,3/16/3,1/16/3,1294.5,1294.6666666666667,[s,stacc])-trailing_score_note.",
            "snote(n5286,[g,#],4,216:5,4/16/3,1/16/3,1294.6666666666667,1294.8333333333333,[s,stacc])-trailing_score_note.",
            "snote(n5287,[a,n],4,216:5,5/16/3,1/16/3,1294.8333333333333,1295.0,[s,stacc])-trailing_score_note.",
            "snote(n5288,[a,#],4,216:6,0,1/16/3,1295.0,1295.1666666666667,[s,stacc])-trailing_score_note.",
            "snote(n5289,[b,n],4,216:6,1/16/3,1/16/3,1295.1666666666667,1295.3333333333333,[s,stacc])-trailing_score_note.",
            "snote(n5290,[b,#],4,216:6,2/16/3,1/16/3,1295.3333333333333,1295.5,[s,stacc])-trailing_score_note.",
            "snote(n5291,[c,#],5,216:6,3/16/3,1/16/3,1295.5,1295.6666666666667,[s,stacc])-trailing_score_note.",
            "snote(n5292,[d,n],5,216:6,4/16/3,1/16/3,1295.6666666666667,1295.8333333333333,[s,stacc])-trailing_score_note.",
            "snote(n5293,[d,#],5,216:6,5/16/3,1/16/3,1295.8333333333333,1296.0,[s,stacc])-trailing_score_note.",
            "snote(n2233,[c,#],4,200:1,2/16,1/8,597.5,598.0,[s])-trailing_score_note.",
            "snote(n2234,[d,n],4,200:2,0,1/8,598.0,598.5,[s])-trailing_score_note.",
            "snote(n2235,[e,n],4,200:2,2/16,1/8,598.5,599.0,[s])-trailing_score_note.",
            "snote(n2236,[f,#],4,200:3,0,1/8,599.0,599.5,[s])-trailing_score_note.",
            "snote(n2237,[g,n],4,200:3,2/16,1/8,599.5,600.0,[s])-trailing_score_note.",
            "snote(n781,[r,-],-,36:3,0,1/4,107.0,108.0,[fermata])-trailing_score_note.",
            "snote(n1304,[r,-],-,45:4,0,1/4,179.0,180.0,[fermata])-trailing_score_note.",
        ]

        for ml in trailing_score_lines:

            mo = MatchSnoteTrailingScoreV0.from_matchline(
                ml,
                version=Version(0, 3, 0),
            )
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchSnoteDeletionV0.from_matchline(ml, version=Version(1, 0, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_insertion_lines_v_0_3_0(self):

        insertion_lines = [
            "insertion-note(178,[e,n],4,42982,43198,43535,5).",
            "insertion-note(411,[b,n],4,98186,98537,98898,1).",
            "insertion-note(583,[e,n],4,128488,129055,129436,12).",
            "insertion-note(599,[c,#],5,130298,130348,130348,62).",
            "insertion-note(603,[c,#],5,130452,130536,130536,68).",
            "insertion-note(604,[b,n],4,130541,130575,130575,63).",
            "insertion-note(740,[e,n],4,148300,149097,149097,6).",
            "insertion-note(756,[c,#],5,150152,150220,150220,72).",
            "insertion-note(759,[c,#],5,150308,150380,150380,70).",
            "insertion-note(761,[b,n],4,150388,150443,150443,71).",
        ]

        for ml in insertion_lines:

            mo = MatchInsertionNoteV0.from_matchline(ml, version=Version(0, 3, 0))

            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

    def test_insertion_lines_v_0_4_0(self):

        insertion_lines = [
            "insertion-note(171,[A,n],5,13216,13248,13248,63).",
            "insertion-note(187,[C,#],5,14089,14132,14132,46).",
            "insertion-note(276,[G,n],4,20555,21144,21144,51).",
            "insertion-note(1038,[E,n],5,70496,70526,70526,55).",
            "insertion-note(1091,[F,#],5,73018,73062,73062,40).",
            "insertion-note(1247,[E,n],2,81885,81920,81920,57).",
            "insertion-note(1252,[F,#],2,82061,82130,82130,17).",
            "insertion-note(1316,[F,#],6,86084,86122,86122,38).",
            "insertion-note(1546,[G,#],1,99495,99536,99536,16).",
            "insertion-note(1557,[B,n],5,100300,100496,100496,80).",
            "insertion-note(1572,[B,n],1,104377,104460,104460,61).",
        ]

        for ml in insertion_lines:

            for minor_version in (4, 5):
                mo = MatchInsertionNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )

                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)

    def test_insertion_lines_v_0_1_0(self):
        insertion_lines = [
            "insertion-note(1,[c,n],6,39060.00,39890.00,38).",
            "insertion-note(6,[c,n],5,48840.00,49870.00,26).",
            "insertion-note(17,[c,n],5,72600.00,75380.00,26).",
            "insertion-note(32,[b,b],5,93030.00,95050.00,32).",
            "insertion-note(85,[b,b],3,162600.00,164950.00,27).",
            "insertion-note(132,[c,n],5,226690.00,227220.00,34).",
            "insertion-note(179,[b,b],4,280360.00,282310.00,35).",
        ]

        for ml in insertion_lines:

            for minor_version in (1, 2):
                mo = MatchInsertionNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )

                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchInsertionNoteV0.from_matchline(ml, version=Version(1, 0, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_hammer_bounce_lines(self):

        # These lines do not exist in any dataset that we have access to
        # for now the tests use lines replacing insertion with hammer_bounce
        hammer_bounce_lines = [
            "hammer_bounce-note(1,[c,n],6,39060.00,39890.00,38).",
            "hammer_bounce-note(6,[c,n],5,48840.00,49870.00,26).",
            "hammer_bounce-note(17,[c,n],5,72600.00,75380.00,26).",
            "hammer_bounce-note(32,[b,b],5,93030.00,95050.00,32).",
            "hammer_bounce-note(85,[b,b],3,162600.00,164950.00,27).",
            "hammer_bounce-note(132,[c,n],5,226690.00,227220.00,34).",
            "hammer_bounce-note(179,[b,b],4,280360.00,282310.00,35).",
        ]

        for ml in hammer_bounce_lines:

            for minor_version in (1, 2):
                mo = MatchHammerBounceNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )

                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)

    def test_trailing_played_lines(self):

        # These lines do not exist in any dataset that we have access to
        # for now the tests use lines replacing insertion with trailing_played_note
        trailing_played_note_lines = [
            "trailing_played_note-note(1,[c,n],6,39060.00,39890.00,38).",
            "trailing_played_note-note(6,[c,n],5,48840.00,49870.00,26).",
            "trailing_played_note-note(17,[c,n],5,72600.00,75380.00,26).",
            "trailing_played_note-note(32,[b,b],5,93030.00,95050.00,32).",
            "trailing_played_note-note(85,[b,b],3,162600.00,164950.00,27).",
            "trailing_played_note-note(132,[c,n],5,226690.00,227220.00,34).",
            "trailing_played_note-note(179,[b,b],4,280360.00,282310.00,35).",
        ]

        for ml in trailing_played_note_lines:

            for minor_version in (1, 2):
                mo = MatchTrailingPlayedNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )

                basic_line_test(mo)
                self.assertTrue(mo.matchline == ml)

    def test_trill_lines(self):

        trill_lines = [
            "trill(1156-1)-note(566,[F,#],5,41161,41246,41246,71).",
            "trill(1158-1)-note(573,[F,#],5,41447,41558,41558,73).",
            "trill(1158-1)-note(574,[F,n],5,41536,41622,41622,63).",
            "trill(1168-1)-note(580,[F,n],5,41876,41976,41976,58).",
            "trill(1168-1)-note(581,[D,#],5,41933,42012,42012,63).",
            "trill(1250-1)-note(657,[F,#],5,47328,47444,47444,71).",
            "trill(1250-1)-note(658,[F,n],5,47384,47482,47482,63).",
            "trill(1252-1)-note(664,[F,n],5,47659,47798,47798,56).",
            "trill(1252-1)-note(666,[D,#],5,47743,47812,47812,59).",
        ]

        for ml in trill_lines:

            mo = MatchTrillNoteV0.from_matchline(ml, version=Version(0, 5, 0))
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchTrillNoteV0.from_matchline(ml, version=Version(1, 0, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_sustain_lines(self):

        sustain_lines = [
            "sustain(711360,22).",
            "sustain(711440,21).",
            "sustain(711520,19).",
            "sustain(711600,18).",
            "sustain(711680,17).",
            "sustain(711760,16).",
            "sustain(711840,16).",
            "sustain(712080,15).",
            "sustain(712560,15).",
            "sustain(715280,14).",
            "sustain(717920,14).",
            "sustain(720080,13).",
            "sustain(721760,13).",
            "sustain(731920,13).",
        ]

        for i, ml in enumerate(sustain_lines):

            mo = MatchSustainPedalV0.from_matchline(ml, version=Version(0, 5, 0))
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchSustainPedalV0.from_matchline(ml, version=Version(1, 0, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_soft_lines(self):

        soft_lines = [
            "soft(2,4).",
            "soft(180,4).",
            "soft(182,3).",
            "soft(184,4).",
            "soft(282,4).",
            "soft(284,3).",
            "soft(286,4).",
            "soft(812,4).",
            "soft(814,3).",
            "soft(816,4).",
            "soft(984,4).",
            "soft(986,3).",
            "soft(988,4).",
            "soft(1006,4).",
            "soft(1008,3).",
        ]

        for i, ml in enumerate(soft_lines):

            mo = MatchSoftPedalV0.from_matchline(ml, version=Version(0, 5, 0))
            basic_line_test(mo)
            self.assertTrue(mo.matchline == ml)

        # An error is raised if parsing the wrong version
        try:
            mo = MatchSoftPedalV0.from_matchline(ml, version=Version(1, 0, 0))
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)


class TestMatchUtils(unittest.TestCase):
    """
    Test utilities for handling match files
    """

    def test_interpret_version(self):
        """
        Test `interpret_version`
        """
        # new version format
        version_string = "1.2.3"

        version = interpret_version(version_string)

        # Check that output is the correct type
        self.assertTrue(isinstance(version, Version))

        # Check that output is correct
        self.assertTrue(
            all(
                [
                    version.major == 1,
                    version.minor == 2,
                    version.patch == 3,
                ]
            )
        )

        # old version format
        version_string = "5.7"

        version = interpret_version(version_string)
        # Check that output is the correct type
        self.assertTrue(isinstance(version, Version))

        # Check that output is correct
        self.assertTrue(
            all(
                [
                    version.major == 0,
                    version.minor == 5,
                    version.patch == 7,
                ]
            )
        )

        # Wrongly formatted version (test that it raises a ValueError)

        version_string = "4.n.9.0"

        try:
            version = interpret_version(version_string)
            # The test should fail if the exception is not raised
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

    def test_fractional_symbolic_duration(self):

        # Test string and float methods and from_string methods
        numerators = RNG.randint(0, 1000, 100)
        denominators = RNG.randint(2, 1000, 100)
        tuple_divs = RNG.randint(0, 5, 100)

        for num, den, td in zip(numerators, denominators, tuple_divs):

            fsd = FractionalSymbolicDuration(num, den, td if td > 0 else None)

            if den > 1 and td > 0:
                expected_string = f"{num}/{den}/{td}"
            elif td == 0:
                expected_string = f"{num}/{den}"

            fsd_from_string = FractionalSymbolicDuration.from_string(expected_string)

            self.assertTrue(fsd_from_string == fsd)
            self.assertTrue(str(fsd) == expected_string)
            self.assertTrue(float(fsd) == num / (den * (td if td > 0 else 1)))

        # The following string should raise an error
        wrong_string = "wrong_string"
        try:
            fsd = FractionalSymbolicDuration.from_string(wrong_string)
            self.assertTrue(False)  # pragma: no cover
        except ValueError:
            self.assertTrue(True)

        # Test bound
        numerators = RNG.randint(0, 10, 100)
        denominators = RNG.randint(1024, 2000, 100)

        for num, den in zip(numerators, denominators):
            fsd = FractionalSymbolicDuration(num, den)
            self.assertTrue(fsd.denominator <= 128)

        # Test addition
        numerators1 = RNG.randint(1, 10, 100)
        denominators1 = RNG.randint(2, 10, 100)
        tuple_divs1 = RNG.randint(0, 5, 100)

        # Test string and float methods
        numerators2 = RNG.randint(0, 10, 100)
        denominators2 = RNG.randint(2, 10, 100)
        tuple_divs2 = RNG.randint(0, 5, 100)

        for num1, den1, td1, num2, den2, td2 in zip(
            numerators1,
            denominators1,
            tuple_divs1,
            numerators2,
            denominators2,
            tuple_divs2,
        ):

            fsd1 = FractionalSymbolicDuration(num1, den1, td1 if td1 > 0 else None)
            fsd2 = FractionalSymbolicDuration(num2, den2, td2 if td2 > 0 else None)
            fsd3_from_radd = FractionalSymbolicDuration(
                num1, den1, td1 if td1 > 0 else None
            )

            fsd3 = fsd1 + fsd2
            fsd3_from_radd += fsd2

            self.assertTrue(fsd3 == fsd3_from_radd)

            if num1 > 0 and num2 > 0:
                self.assertTrue(str(fsd3) == f"{str(fsd1)}+{str(fsd2)}")
                # Test allow_additions option in from_string method
                fsd_from_string = FractionalSymbolicDuration.from_string(
                    str(fsd3), allow_additions=True
                )
                self.assertTrue(fsd_from_string == fsd3)
                # check addition when the two of them have add_components
                fsd3_t_2 = fsd3 + fsd3_from_radd
                self.assertTrue(2 * float(fsd3) == float(fsd3_t_2))

                # check_addition when only one of them has add_components
                fsd4 = fsd1 + fsd3

                self.assertTrue(np.isclose(float(fsd4), float(fsd1) + float(fsd3)))

                fsd3_from_radd += fsd1
                self.assertTrue(
                    np.isclose(float(fsd3_from_radd), float(fsd1) + float(fsd3))
                )

                # They must be different because the order of the
                # additive components would be inverted
                self.assertTrue(fsd3_from_radd != fsd4)

            elif num1 > 0:
                self.assertTrue(str(fsd3) == str(fsd1))

            self.assertTrue(isinstance(fsd3, FractionalSymbolicDuration))
            self.assertTrue(np.isclose(float(fsd1) + float(fsd2), float(fsd3)))

    def test_match_time_signature(self):
        """
        Test MatchTimeSignature
        """
        # These lines were taken from match files and
        # represent all different versions of encoding
        # time signature accross all datasets.
        lines = [
            (
                "[3/4,2/4,9/8,2/2,3/4,9/8,4/4,3/4]",
                3,
                4,
                "[3/4,2/4,9/8,2/2,3/4,9/8,4/4,3/4]",
            ),
            ("[3/4]", 3, 4, "[3/4]"),
            ("3/4", 3, 4, "[3/4]"),
            ("[6/8,12/16,6/8]", 6, 8, "[6/8,12/16,6/8]"),
        ]

        for line, num, den, ll in lines:

            ts = MatchTimeSignature.from_string(line)

            self.assertTrue(ts.numerator == num)
            self.assertTrue(ts.denominator == den)
            ts.is_list = True
            self.assertTrue(str(ts) == ll)
            ts.is_list = False
            self.assertTrue(str(ts) == f"{num}/{den}")

    def test_to_camel_case(self):

        snake_case = "this_is_a_string"
        expected_string = "thisIsAString"

        self.assertTrue(to_camel_case(snake_case) == expected_string)

    def test_to_snake_case(self):

        camel_case = "thisIsAString"
        expected_string = "this_is_a_string"
        self.assertTrue(to_snake_case(camel_case) == expected_string)

    def test_match_key_signature(self):

        lines = [
            "[fn,major]",
            "[en,minor]",
            "[A Maj/F# min]",
            "[Ab Maj/F min,Db Maj/Bb min]",
            "[B Maj/G# min,A Maj/F# min,B Maj/G# min]",
            "B Maj",
            "[Db Maj/Bb min,A Maj/F# min,Db Maj/Bb min,A Maj/F# min,Db Maj/Bb min]",
        ]

        for line in lines:
            ks = MatchKeySignature.from_string(line)

            if line.startswith("[") and ks.fmt == "v0.3.0":
                ks.is_list = True
            else:
                ks.is_list = False
            ks_str = str(ks)
            self.assertTrue(ks_str == line)

            ks.fmt = "v1.0.0"
            for component in ks.other_components:
                key_name = fifths_mode_to_key_name(component.fifths, component.mode)
                self.assertTrue(str(component).startswith(key_name))

if __name__ == "__main__":
    unittest.main()
