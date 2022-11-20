#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Matchfile import
"""
import unittest
import numpy as np

from tests import MATCH_IMPORT_EXPORT_TESTFILES, MOZART_VARIATION_FILES

from partitura.io.matchlines_v1 import (
    MatchInfo,
    MatchScoreProp,
    MatchSection,
    MatchSnote,
    MatchNote,
)

from partitura.io.matchfile_base import interpret_version, Version, MatchError


class TestMatchLinesV1_0_0(unittest.TestCase):
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
        midiClockUnits_line = "info(midiClockUnits,4000)."
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

        for ml in matchlines:
            mo = MatchInfo.from_matchline(ml)
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            self.assertTrue(mo.matchline == ml)

            # assert that the data types of the match line are correct
            self.assertTrue(mo.check_types())

        # The following lines should result in an error
        try:
            # This line is not defined as an info line and should raise an error
            notSpecified_line = "info(notSpecified,value)."

            mo = MatchInfo.from_matchline(notSpecified_line)
            self.assertTrue(False)
        except ValueError:
            # assert that the error was raised
            self.assertTrue(True)

        try:
            # wrong value (string instead of integer)
            midiClockUnits_line = "info(midiClockUnits,wrong_value)."

            mo = MatchInfo.from_matchline(midiClockUnits_line)
            self.assertTrue(False)
        except ValueError:
            # assert that the error was raised
            self.assertTrue(True)

        try:
            # This is not a valid line and should result in a MatchError
            wrong_line = "wrong_line"
            mo = MatchInfo.from_matchline(wrong_line)
            self.assertTrue(False)
        except MatchError:
            self.assertTrue(True)

    def test_score_prop_lines(self):

        keysig_line = "scoreprop(keySignature,E,0:2,1/8,-0.5000)."

        timesig_line = "scoreprop(timeSignature,2/4,0:2,1/8,-0.5000)."

        directions_line = "scoreprop(directions,[Allegro],0:2,1/8,-0.5000)."

        beatsubdivision_line = "scoreprop(beatSubDivision,2,0:2,1/8,-0.5000)."

        matchlines = [
            keysig_line,
            timesig_line,
            directions_line,
            beatsubdivision_line,
        ]

        for ml in matchlines:
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            mo = MatchScoreProp.from_matchline(ml)
            self.assertTrue(mo.matchline == ml)

            # assert that the data types of the match line are correct
            self.assertTrue(mo.check_types())

        try:
            # This is not a valid line and should result in a MatchError
            wrong_line = "wrong_line"
            mo = MatchScoreProp.from_matchline(wrong_line)
            self.assertTrue(False)
        except MatchError:
            self.assertTrue(True)

    def test_section_lines(self):

        section_lines = [
            "section(0.0000,100.0000,0.0000,100.0000,[end]).",
            "section(100.0000,200.0000,0.0000,100.0000,[fine]).",
            "section(100.0000,200.0000,0.0000,100.0000,[volta end]).",
            "section(100.0000,200.0000,0.0000,100.0000,[repeat left]).",
        ]

        for ml in section_lines:
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            mo = MatchSection.from_matchline(ml)
            # print(mo.matchline, ml, [(g == t, g, t) for g, t in zip(mo.matchline, ml)])
            self.assertTrue(mo.matchline == ml)

            # assert that the data types of the match line are correct
            self.assertTrue(mo.check_types())

        # Check version (an error should be raised for old versions)
        try:
            mo = MatchSection.from_matchline(section_lines[0], version=Version(0, 5, 0))
            self.assertTrue(False)

        except ValueError:
            self.assertTrue(True)

        # Check that incorrectly formatted line results in a match error
        try:
            # Line does not have [] for the end annotations
            wrong_line = "section(0.0000,100.0000,0.0000,100.0000,end)."
            mo = MatchSection.from_matchline(wrong_line)
            self.assertTrue(False)
        except MatchError:
            self.assertTrue(True)

    def test_snote_lines(self):

        snote_lines = [
            "snote(n1,[B,n],3,0:2,1/8,1/8,-0.5000,0.0000,[v1])",
            "snote(n3,[G,#],3,1:1,0,1/16,0.0000,0.2500,[v3])",
            "snote(n1,[E,n],4,1:1,0,1/4,0.0000,1.0000,[arp])",
            "snote(n143,[B,b],5,7:2,2/16,1/8,25.5000,26.0000,[s,stacc])",
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
        ]

        for ml, strl in zip(snote_lines, output_strings):
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            mo = MatchSnote.from_matchline(ml)
            # test __str__ method
            self.assertTrue(
                all(
                    [
                        ll.strip() == sl.strip()
                        for ll, sl in zip(str(mo).splitlines(), strl.splitlines())
                    ]
                )
            )
            # print(mo.matchline, ml)
            self.assertTrue(mo.matchline == ml)

            # assert that the data types of the match line are correct
            self.assertTrue(mo.check_types())

        try:
            # This is not a valid line and should result in a MatchError
            wrong_line = "wrong_line"
            mo = MatchSnote.from_matchline(wrong_line)
            self.assertTrue(False)
        except MatchError:
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
            mo = MatchNote.from_matchline(ml)
            # print(mo.matchline, ml, [(g == t, g, t) for g, t in zip(mo.matchline, ml)])
            self.assertTrue(mo.matchline == ml)

            # assert that the data types of the match line are correct
            self.assertTrue(mo.check_types())


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
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
