#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the key estimation methods.
"""
import unittest

from partitura import EXAMPLE_MUSICXML
from partitura import load_score
from partitura.score import Interval
from partitura.utils.music import transpose
import numpy as np


class TransposeScoreByInterval(unittest.TestCase):
    def test_transpose(self):
        score = load_score(EXAMPLE_MUSICXML)
        interval = Interval(number=5, quality="d")
        new_score = transpose(score, interval)
        note_array = new_score.note_array(include_pitch_spelling=True)
        steps = np.array(["E", "G", "B"])
        alters = np.array([-1, -1, -1])
        octaves = np.array([5, 5, 5])
        self.assertEqual(np.all(steps == note_array["step"]), True)
        self.assertEqual(np.all(alters == note_array["alter"]), True)
        self.assertEqual(np.all(octaves == note_array["octave"]), True)
