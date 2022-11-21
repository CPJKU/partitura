#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the pitch spelling algorithms.
"""
import numpy as np
import unittest

from partitura import EXAMPLE_MUSICXML
from partitura import load_musicxml
from partitura.musicanalysis import estimate_spelling


def compare_spelling(spelling, notes):
    comparisons = np.zeros((len(spelling), 3))
    for i, (n, s) in enumerate(zip(notes, spelling)):
        comparisons[i, 0] = int(n.step == s["step"])
        if n.alter is None and s["alter"] == 0:
            comparisons[i, 1] = 1
        else:
            comparisons[i, 1] = int(n.alter == s["alter"])
        comparisons[i, 2] = int(n.octave == s["octave"])
    return comparisons


class TestKeyEstimation(unittest.TestCase):
    """
    Test key estimation
    """

    score = load_musicxml(EXAMPLE_MUSICXML)

    def test_part(self):
        spelling = estimate_spelling(self.score[0])
        comparisons = compare_spelling(spelling, self.score[0].notes)
        self.assertTrue(np.all(comparisons), "Incorrect spelling")

    def test_note_array(self):
        spelling = estimate_spelling(self.score[0].note_array())
        comparisons = compare_spelling(spelling, self.score[0].notes)
        self.assertTrue(np.all(comparisons), "Incorrect spelling")
