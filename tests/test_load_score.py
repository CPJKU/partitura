#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for the `load_score` method.
"""
import unittest

from tests import (
    MUSICXML_IMPORT_EXPORT_TESTFILES,
    MEI_TESTFILES,
    KERN_TESTFILES,
    MATCH_IMPORT_EXPORT_TESTFILES,
)


from partitura import (
    load_score,
    EXAMPLE_MIDI,
    EXAMPLE_KERN,
    EXAMPLE_MEI,
    EXAMPLE_MUSICXML,
)
from partitura.io import NotSupportedFormatError
from partitura.score import Part, PartGroup, Score


EXAMPLE_FILES = [EXAMPLE_MIDI, EXAMPLE_KERN, EXAMPLE_MEI, EXAMPLE_MUSICXML]


class TestLoadScore(unittest.TestCase):
    def test_load_score(self):

        for fn in (
            MUSICXML_IMPORT_EXPORT_TESTFILES
            + MEI_TESTFILES
            + KERN_TESTFILES
            + MATCH_IMPORT_EXPORT_TESTFILES
            + EXAMPLE_FILES
        ):
            self.check_return_type(fn)

    def check_return_type(self, fn):
        score = load_score(fn)
        self.assertTrue(isinstance(score, Score), f"results of load_score type are not Score for score {fn}.")
        for pp in score.part_structure:
            self.assertTrue(type(pp) in (Part, PartGroup), f"results of score.part_structure type are neither Part or PartGroup for score {fn}.")
        for pp in score.parts:
            self.assertTrue(isinstance(pp, Part), f"results of score.parts type are not Part for score {fn}.")
