#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains test functions for MEI import
"""

import unittest

from tests import MUSESCORE_TESTFILES
from partitura import load_musicxml, load_mei, EXAMPLE_MEI
import partitura.score as score
from partitura.io.importmei import MeiParser
from partitura.utils import compute_pianoroll
from lxml import etree
from xmlschema.names import XML_NAMESPACE
from partitura.io import load_score, load_via_musescore
from partitura.io.musescore import find_musescore, MuseScoreNotFoundException
import platform

import numpy as np
from pathlib import Path

try:
  if find_musescore():
    class TestImportMusescore(unittest.TestCase):
        def test_epfl_scores(self):
            score = load_via_musescore(MUSESCORE_TESTFILES[0])
            self.assertTrue(len(score.parts) == 1)
            # try the generic loading function
            score = load_score(MUSESCORE_TESTFILES[0])
            self.assertTrue(len(score.parts) == 1)
except MuseScoreNotFoundException:
    pass
