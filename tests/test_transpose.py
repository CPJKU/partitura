#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the key estimation methods.
"""
import unittest

from partitura import EXAMPLE_MUSICXML
from partitura import load_score
from partitura.score import Interval, KeySignature
from partitura.utils.music import transpose
import numpy as np


class TransposeScoreByInterval(unittest.TestCase):
    def test_transpose(self):
        score = load_score(EXAMPLE_MUSICXML)

        # Test transposing up a diminished fifth
        # A4 C5 E5 -> Eb5 Gb5 Bb5 (from C to Gb key)
        interval = Interval(number=5, quality="d", direction="up")
        new_score = transpose(score, interval, transpose_key_signatures=True)
        note_array = new_score.note_array(include_pitch_spelling=True)
        steps = np.array(["E", "G", "B"])
        alters = np.array([-1, -1, -1])
        octaves = np.array([5, 5, 5])
        self.assertEqual(np.all(steps == note_array["step"]), True)
        self.assertEqual(np.all(alters == note_array["alter"]), True)
        self.assertEqual(np.all(octaves == note_array["octave"]), True)
        self.assertEqual(next(new_score[0].iter_all(KeySignature)).name, "Gb")

        # Test transposing down a diminished fifth
        # A4 C5 E5 -> D#4 F#4 A#4 (from C to F# key)
        interval = Interval(number=5, quality="d", direction="down")
        new_score = transpose(score, interval, transpose_key_signatures=True)
        note_array = new_score.note_array(include_pitch_spelling=True)
        steps = np.array(["D", "F", "A"])
        alters = np.array([1, 1, 1])
        octaves = np.array([4, 4, 4])
        self.assertEqual(np.all(steps == note_array["step"]), True)
        self.assertEqual(np.all(alters == note_array["alter"]), True)
        self.assertEqual(np.all(octaves == note_array["octave"]), True)
        self.assertEqual(next(new_score[0].iter_all(KeySignature)).name, "F#")
