#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for part properties.
"""
import unittest
import partitura
from partitura import score


class TestingPartProperties(unittest.TestCase):
    def test_props(self):
        part = score.Part("P0", "My Part")
        part.set_quarter_duration(0, 10)
        part.add(score.KeySignature(0, ""), start=0)
        part.add(score.TimeSignature(3, 4), start=0)
        part.add(score.TimeSignature(4, 4), start=30)
        part.add(score.Note(id="n0", step="A", octave=4), start=0, end=10)
        part.add(score.Rest(id="r0"), start=10, end=20)
        part.add(score.Note(id="n1", step="A", octave=4), start=20, end=70)
        part.add(score.ImpulsiveLoudnessDirection("sfz", "sfz"), start=20, end=20)
        part.add(score.ArticulationDirection("staccato", "staccato"), start=0, end=0)
        score.add_measures(part)
        score.tie_notes(part)

        self.assertTrue(len(part.notes) == 3)
        self.assertTrue(len(part.notes_tied) == 2)
        self.assertTrue(len(part.measures) == 2)
        self.assertTrue(len(part.time_sigs) == 2)
        self.assertTrue(len(part.key_sigs) == 1)
        self.assertTrue(len(part.rests) == 1)
        self.assertTrue(len(part.articulations) == 1)
        self.assertTrue(len(part.dynamics) == 1)
        self.assertTrue(len(part.repeats) == 0)


if __name__ == "__main__":
    unittest.main()
