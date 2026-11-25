#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for splitting and tying notes
"""
import unittest

import partitura.score as score


class TestTieNotes(unittest.TestCase):
    """
    Test `tie_notes`
    """

    def make_part(self, *notes, n_measures=2):
        """
        Make a part in 4/4 with notes given by onset and offset in 16th notes
        """
        part = score.Part("P1", quarter_duration=4)
        part.add(score.TimeSignature(4, 4), 0)
        for i in range(n_measures):
            part.add(score.Measure(number=i + 1), 16 * i, 16 * (i + 1))
        for i, (onset, offset) in enumerate(notes):
            note = score.Note("C", 4, id=f"n{i}", voice=1, staff=1)
            part.add(note, onset, offset)
        return part

    def get_notes(self, part):
        return [
            (n.start.t, n.end.t, n.symbolic_duration["type"], n.symbolic_duration.get("dots", 0))
            for n in part.notes
        ]

    def test_composite_duration(self):
        """
        Composite durations should be split, since they cannot be notated
        """
        for duration, expected in (
            # 5/16 and 5/8
            (5, [(0, 4, "quarter", 0), (4, 5, "16th", 0)]),
            (10, [(0, 8, "half", 0), (8, 10, "eighth", 0)]),
        ):
            part = self.make_part((0, duration))
            score.tie_notes(part)
            self.assertEqual(self.get_notes(part), expected)

            first, second = part.notes
            self.assertIs(first.tie_next, second)
            self.assertIs(second.tie_prev, first)

            # this is still the same note
            na = part.note_array()
            self.assertEqual(list(na["id"]), ["n0"])
            self.assertEqual(list(na["duration_div"]), [duration])

    def test_composite_duration_across_barline(self):
        """
        Notes are split at barlines, and the resulting notes if necessary
        """
        # 10 sixteenths in both measures
        part = self.make_part((6, 26))
        score.tie_notes(part)
        notes = part.notes
        self.assertGreater(len(notes), 2)
        self.assertEqual(notes[0].start.t, 6)
        self.assertEqual(notes[-1].end.t, 26)
        for note, next_note in zip(notes[:-1], notes[1:]):
            self.assertEqual(note.end.t, next_note.start.t)
            self.assertIs(note.tie_next, next_note)
            self.assertIs(next_note.tie_prev, note)
            self.assertTrue(note.symbolic_duration)
        self.assertEqual(list(part.note_array()["duration_div"]), [20])

    def test_simple_duration(self):
        """
        Other notes should not be changed
        """
        part = self.make_part((0, 4), (4, 10), (10, 11), (16, 32))
        notes = self.get_notes(part)
        score.tie_notes(part)
        self.assertEqual(self.get_notes(part), notes)
        self.assertTrue(all(n.tie_next is None for n in part.notes))


if __name__ == "__main__":
    unittest.main()
