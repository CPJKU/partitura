#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Matchfile import
"""
import unittest
import numpy as np

from tests import MATCH_IMPORT_EXPORT_TESTFILES, MOZART_VARIATION_FILES

from partitura.io.matchlines_v1 import (
    MatchInfo as MatchInfoV1,
    MatchScoreProp as MatchScorePropV1,
    MatchSection as MatchSectionV1,
    MatchSnote as MatchSnoteV1,
    MatchNote as MatchNoteV1,
    MatchSnoteNote as MatchSnoteNoteV1,
)

from partitura.io.matchlines_v0 import (
    MatchInfo as MatchInfoV0,
    MatchSnote as MatchSnoteV0,
    MatchNote as MatchNoteV0,
)

from partitura.io.matchfile_base import MatchError

from partitura.io.matchfile_utils import (
    FractionalSymbolicDuration,
    Version,
    interpret_version,
)

RNG = np.random.RandomState(1984)


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

        for i, ml in enumerate(matchlines):
            mo = MatchInfoV1.from_matchline(ml)
            # assert that the information from the matchline
            # is parsed correctly and results in an identical line
            # to the input match line
            self.assertTrue(mo.matchline == ml)

            # assert that the data types of the match line are correct
            if i == 0:
                # Test verbose output
                self.assertTrue(mo.check_types(verbose=True))
            else:
                self.assertTrue(mo.check_types())

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
            # This is not a valid line and should result in a MatchError
            wrong_line = "wrong_line"
            mo = MatchInfoV1.from_matchline(wrong_line)
            self.assertTrue(False)  # pragma: no cover
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
            mo = MatchScorePropV1.from_matchline(ml)
            self.assertTrue(mo.matchline == ml)

            # assert that the data types of the match line are correct
            self.assertTrue(mo.check_types())

        try:
            # This is not a valid line and should result in a MatchError
            wrong_line = "wrong_line"
            mo = MatchScorePropV1.from_matchline(wrong_line)
            self.assertTrue(False)  # pragma: no cover
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
            mo = MatchSectionV1.from_matchline(ml)

            self.assertTrue(mo.matchline == ml)

            # assert that the data types of the match line are correct
            self.assertTrue(mo.check_types())

        # Check version (an error should be raised for old versions)
        try:
            mo = MatchSectionV1.from_matchline(
                section_lines[0], version=Version(0, 5, 0)
            )
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
            mo = MatchSnoteV1.from_matchline(ml)
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

            self.assertTrue(mo.MidiPitch < 128)

            # assert that the data types of the match line are correct
            self.assertTrue(mo.check_types())

        try:
            # This is not a valid line and should result in a MatchError
            wrong_line = "wrong_line"
            mo = MatchSnoteV1.from_matchline(wrong_line)
            self.assertTrue(False)  # pragma: no cover
        except MatchError:
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
            mo = MatchNoteV1.from_matchline(ml)
            # print(mo.matchline, ml, [(g == t, g, t) for g, t in zip(mo.matchline, ml)])
            self.assertTrue(mo.matchline == ml)

            # assert that the data types of the match line are correct
            self.assertTrue(mo.check_types())

    def test_snotenote_lines(self):

        snotenote_lines = [
            "snote(n1,[B,n],3,0:2,1/8,1/8,-0.5000,0.0000,[1])-note(0,47,39940,42140,44,0,0).",
        ]

        for i, ml in enumerate(snotenote_lines):
            # snote = MatchSnoteV1.from_matchline(ml)
            # note = MatchNoteV1.from_matchline(ml)

            mo = MatchSnoteNoteV1.from_matchline(ml, version=Version(1, 0, 0))

            self.assertTrue(mo.matchline == ml)

            if i == 0:
                self.assertTrue(mo.check_types(verbose=True))
            else:
                self.assertTrue(mo.check_types())


class TestMatchLinesV0(unittest.TestCase):
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
                # print(mo.matchline, ml)
                self.assertTrue(mo.matchline == ml)

                # assert that the data types of the match line are correct
                self.assertTrue(mo.check_types())

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
                # print(mo.matchline, ml)
                self.assertTrue(mo.matchline == ml)

                # assert that the data types of the match line are correct
                self.assertTrue(mo.check_types())

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
                # print(mo.matchline, ml)
                self.assertTrue(mo.matchline == ml)

                # assert that the data types of the match line are correct
                self.assertTrue(mo.check_types())

    def test_note_lines_v_0_3_0(self):

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

            for minor_version in range(3, 6):
                mo = MatchNoteV0.from_matchline(
                    ml, version=Version(0, minor_version, 0)
                )
                self.assertTrue(mo.matchline == ml)
                # check duration and adjusted duration
                self.assertTrue(mo.AdjDuration >= mo.Duration)

                # assert that the data types of the match line are correct
                self.assertTrue(mo.check_types())

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
            elif num2 > 0:
                self.assertTrue(str(fsd3) == str(fsd2))

            self.assertTrue(isinstance(fsd3, FractionalSymbolicDuration))
            self.assertTrue(np.isclose(float(fsd1) + float(fsd2), float(fsd3)))
