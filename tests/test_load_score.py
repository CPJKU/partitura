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
            self.load_score(fn)

    def load_score(self, fn):
        try:
            score = load_score(fn)
            self.assertTrue(isinstance(score, Score))

            for pp in score.part_structure:
                self.assertTrue(type(pp) in (Part, PartGroup))
            for pp in score.parts:
                self.assertTrue(isinstance(pp, Part))
        except NotSupportedFormatError:
            self.assertTrue(False)
