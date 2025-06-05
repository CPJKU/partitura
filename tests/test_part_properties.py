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

    def test_removing_points(self):
        part = score.Part("P0", "My Part")
        part.set_quarter_duration(0, 10)
        part.add((ts0 := score.TimeSignature(3, 4)), start=0)
        part.add((ts1 := score.TimeSignature(4, 4)), start=20)
        part.add((n0 := score.Note(id="n0", step="A", octave=4)), start=0, end=10)
        part.add((r0 := score.Rest(id="r0")), start=10, end=20)
        part.add((n1 := score.Note(id="n1", step="A", octave=4)), start=20, end=30)

        # Part id="P0" name="My Part"
        #  ├─ TimePoint t=0 quarter=10
        #  │   └─ starting objects
        #  │       ├─ 0--10 Note id=n0 voice=None staff=None type=quarter pitch=A4
        #  │       └─ 0-- TimeSignature 3/4
        #  ├─ TimePoint t=10 quarter=10
        #  │   ├─ ending objects
        #  │   │   └─ 0--10 Note id=n0 voice=None staff=None type=quarter pitch=A4
        #  │   └─ starting objects
        #  │       └─ 10--20 Rest id=r0 voice=None staff=None type=quarter
        #  ├─ TimePoint t=20 quarter=10
        #  │   ├─ ending objects
        #  │   │   └─ 10--20 Rest id=r0 voice=None staff=None type=quarter
        #  │   └─ starting objects
        #  │       ├─ 20--30 Note id=n1 voice=None staff=None type=quarter pitch=A4
        #  │       └─ 20-- TimeSignature 4/4
        #  └─ TimePoint t=30 quarter=10
        #      └─ ending objects
        #          └─ 20--30 Note id=n1 voice=None staff=None type=quarter pitch=A4

        # Points: [0, 10, 20, 30]
        p0, p10, p20, p30 = part._points

        # Removing n0 should not affect the score (start and end point still needed)
        num_points = len(part._points)
        part.remove(n0)
        self.assertTrue(len(part._points) == num_points)

        # Removing r0 should remove the point 10
        part.remove(r0)
        self.assertTrue(len(part._points) == num_points - 1)
        self.assertTrue(p0.next is p20)
        self.assertTrue(p20.prev is p0)

        # Removing n1 should remove the last point
        part.remove(n1)
        self.assertTrue(len(part._points) == num_points - 2)
        self.assertTrue(part.last_point is p20)
        self.assertTrue(p20.next is None)

        # Removing the first TS should set p20 to the first point
        part.remove(ts0)
        self.assertTrue(part.first_point is p20)
        self.assertTrue(p20.prev is None)

        # Removing all but one point should result in prev/next being None
        self.assertTrue(len(part._points) == 1)
        self.assertTrue(part.first_point is part.last_point)
        self.assertTrue(part.first_point.prev is None)
        self.assertTrue(part.first_point.next is None)

        # Removing last point should not error
        part.remove(ts1)
        self.assertTrue(len(part._points) == 0)


if __name__ == "__main__":
    unittest.main()
