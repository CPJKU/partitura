#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for io of the harmony musicxml tag.
"""

import unittest
from partitura import load_musicxml
from partitura.score import ChordSymbol, RomanNumeral
from tests import HARMONY_TESTFILES


class HarmonyImportTester(unittest.TestCase):
    part = load_musicxml(HARMONY_TESTFILES[0])[0]
    def test_chordsymbol(self):
        roots = list()
        kinds = list()
        for cs in self.part.iter_all(ChordSymbol):
            roots.append(cs.root)
            kinds.append(cs.kind)
        self.assertEqual(roots, ['C', 'G'])
        self.assertEqual(kinds, ['m', '7'])

    def test_romanNumeral(self):
        text = list()
        for cs in self.part.iter_all(RomanNumeral):
            text.append(cs.text)
        self.assertEqual(text, ['i', 'V7'])



