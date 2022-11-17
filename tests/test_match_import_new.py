#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Matchfile import
"""
import unittest
import numpy as np

from tests import MATCH_IMPORT_EXPORT_TESTFILES, MOZART_VARIATION_FILES

from partitura.io.matchlines_1_0_0 import (
    MatchInfo,
    MatchScoreProp,
)

from partitura.io.matchfile_base import interpret_version, Version


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

    def test_score_prop_lines(self):

        keysig_line = "scoreprop(keySignature,E,0:2,1/8,-0.5000)."

        timesig_line = "scoreprop(timeSignature,2/4,0:2,1/8,-0.5000)."

        directions_line = "scoreprop(directions,[Allegro],0:2,1/8,-0.5000)."

        

        matchlines = [
            keysig_line,
            timesig_line,
            directions_line,
        ]

        for ml in matchlines:
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            mo = MatchScoreProp.from_matchline(ml)
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
