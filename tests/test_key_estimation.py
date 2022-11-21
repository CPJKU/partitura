#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the key estimation methods.
"""
import unittest

from partitura import EXAMPLE_MUSICXML
from partitura import load_musicxml
from partitura.musicanalysis import estimate_key
from partitura.utils import key_name_to_fifths_mode, fifths_mode_to_key_name


class KeyModeComputation(unittest.TestCase):
    minor_sharps = ["Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m"]
    minor_flats = ["Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm", "Abm"]
    major_sharps = ["G", "D", "A", "E", "B", "F#", "C#"]
    major_flats = ["F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"]

    def test_key_name_to_fifths_mode(self):
        self.assertEqual((0, "minor"), key_name_to_fifths_mode("Am"))
        self.assertEqual((0, "major"), key_name_to_fifths_mode("C"))
        [
            self.assertEqual((i + 1, "minor"), key_name_to_fifths_mode(a))
            for i, a in enumerate(self.minor_sharps)
        ]
        [
            self.assertEqual((-(i + 1), "minor"), key_name_to_fifths_mode(a))
            for i, a in enumerate(self.minor_flats)
        ]
        [
            self.assertEqual((i + 1, "major"), key_name_to_fifths_mode(a))
            for i, a in enumerate(self.major_sharps)
        ]
        [
            self.assertEqual((-(i + 1), "major"), key_name_to_fifths_mode(a))
            for i, a in enumerate(self.major_flats)
        ]

    def test_fifths_modes_to_key_name(self):
        self.assertEqual("Am", fifths_mode_to_key_name(0, "minor"))
        self.assertEqual("C", fifths_mode_to_key_name(0, "major"))
        [
            self.assertEqual(a, fifths_mode_to_key_name(i + 1, "minor"))
            for i, a in enumerate(self.minor_sharps)
        ]
        [
            self.assertEqual(a, fifths_mode_to_key_name(-(i + 1), "minor"))
            for i, a in enumerate(self.minor_flats)
        ]
        [
            self.assertEqual(
                a,
                fifths_mode_to_key_name(i + 1, "major"),
            )
            for i, a in enumerate(self.major_sharps)
        ]
        [
            self.assertEqual(a, fifths_mode_to_key_name(-(i + 1), "major"))
            for i, a in enumerate(self.major_flats)
        ]


class TestKeyEstimation(unittest.TestCase):
    """
    Test key estimation
    """

    score = load_musicxml(EXAMPLE_MUSICXML)

    def test_part(self):
        key = estimate_key(self.score)
        self.assertTrue(key == "Am", "Incorrect key")

    def test_note_array(self):
        key = estimate_key(self.score.note_array())
        self.assertTrue(key == "Am", "Incorrect key")


if __name__ == "__main__":
    unittest.main()
