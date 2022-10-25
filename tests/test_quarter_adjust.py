#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for adjusting quarter durations
"""

import logging
import unittest

import partitura.score as score

LOGGER = logging.getLogger(__name__)

import partitura
import partitura.score as score

# class TestQuarterAdjust(unittest.TestCase):
class TestQuarterAdjust(object):
    """
    Test whether timepoints are adjusted correctly when calling
    set_quarter_duration with the option adjust_times=True.
    """

    cases = [
        dict(
            q_times=[0],
            q_durs=[10],
            note_ons=[0, 10, 20],
            note_offs=[10, 20, 30],
            new_q_time=[0],
            new_q_dur=[100],
            new_note_ons=[0, 100, 200],
            new_note_offs=[100, 200, 300],
        ),
        dict(
            q_times=[0, 10],
            q_durs=[10, 100],
            note_ons=[0, 10, 110],
            note_offs=[10, 110, 210],
            new_q_time=[5, 10],
            new_q_dur=[20, 200],
            new_note_ons=[0, 15, 215],
            new_note_offs=[15, 215, 415],
        ),
        dict(
            q_times=[0, 10],
            q_durs=[10, 100],
            note_ons=[0, 10, 110],
            note_offs=[10, 110, 210],
            new_q_time=[300, 1000],
            new_q_dur=[1000, 2000],
            new_note_ons=[0, 10, 110],
            new_note_offs=[10, 110, 210],
        ),
        dict(
            q_times=[0, 10],
            q_durs=[10, 100],
            note_ons=[0, 10, 110],
            note_offs=[10, 110, 210],
            new_q_time=[5, 10],
            new_q_dur=[20, 200],
            new_note_ons=[0, 15, 215],
            new_note_offs=[15, 215, 415],
        ),
    ]

    def do_test(
        self,
        q_times,
        q_durs,
        note_ons,
        note_offs,
        new_q_time,
        new_q_dur,
        new_note_ons,
        new_note_offs,
    ):

        part = score.Part("P0")

        for q_time, q_dur in zip(q_times, q_durs):
            part.set_quarter_duration(q_time, q_dur)

        for i, (note_on, note_off) in enumerate(zip(note_ons, note_offs)):
            n = score.Note(id="n{}".format(i), step="C", octave=4, voice=1)
            part.add(n, note_on, note_off)

        for q_time, q_dur in zip(new_q_time[::-1], new_q_dur[::-1]):
            part.set_quarter_duration(q_time, q_dur, adjust_times=True)

        for n, start, end in zip(part.notes, new_note_ons, new_note_offs):
            msg = "Note onset {} should be {}".format(n.start.t, start)
            self.assertEqual(n.start.t, start, msg)
            msg = "Note offset {} should be {}".format(n.end.t, end)
            self.assertEqual(n.end.t, end, msg)

    def test_cases(self):
        for test_case in self.cases:
            self.do_test(**test_case)


# class TestQuarterMultiply(unittest.TestCase):
class TestQuarterMultiply(object):
    """
    Test whether timepoints are adjusted correctly when calling
    set_quarter_duration with the option adjust_times=True.
    """

    cases = [
        dict(
            q_times=[0],
            q_durs=[10],
            note_ons=[0, 10, 20],
            note_offs=[10, 20, 30],
            factor=100,
            new_note_ons=[0, 1000, 2000],
            new_note_offs=[1000, 2000, 3000],
        ),
        dict(
            q_times=[0, 10],
            q_durs=[10, 100],
            note_ons=[0, 10, 110],
            note_offs=[10, 110, 210],
            factor=200,
            new_note_ons=[0, 2000, 22000],
            new_note_offs=[2000, 22000, 42000],
        ),
    ]

    def do_test(
        self, q_times, q_durs, note_ons, note_offs, factor, new_note_ons, new_note_offs
    ):

        part = score.Part("P0")

        for q_time, q_dur in zip(q_times, q_durs):
            part.set_quarter_duration(q_time, q_dur)

        for i, (note_on, note_off) in enumerate(zip(note_ons, note_offs)):
            n = score.Note(id="n{}".format(i), step="C", octave=4, voice=1)
            part.add(n, note_on, note_off)

        part.multiply_quarter_durations(factor)

        for n, start, end in zip(part.notes, new_note_ons, new_note_offs):
            msg = "Note {} onset {} should be {}".format(n.id, n.start.t, start)
            self.assertEqual(n.start.t, start, msg)
            msg = "Note {} offset {} should be {}".format(n.id, n.end.t, end)
            self.assertEqual(n.end.t, end, msg)

    def test_cases(self):
        for test_case in self.cases:
            self.do_test(**test_case)


if __name__ == "__main__":
    unittest.main()
