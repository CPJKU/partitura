#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for testing conversions from beats and quarters.
"""
import unittest

import partitura.score as score
import partitura


def test_time_pairs(part, time_pairs):
    bm = part.beat_map
    ibm = part.inv_beat_map
    qm = part.quarter_map
    iqm = part.inv_quarter_map
    for tb, tq in time_pairs:
        assert qm(ibm(tb)) == tq
        assert bm(iqm(tq)) == tb


class TestBeatVsQuarterTimes(unittest.TestCase):
    def test_times_1(self):
        # 4/4 anacrusis
        part = score.Part("id")
        # 1 div is 1 quarter
        part.set_quarter_duration(0, 1)
        # 4/4 at t=0
        part.add(score.TimeSignature(4, 4), 0)

        # ANACRUSIS
        # quarter note from t=0 to t=1
        part.add(score.Note("c", 4), 0, 1)
        # incomplete measure from t=0 to t=1
        part.add(score.Measure(), 0, 1)

        # whole note from t=1 to t=5
        part.add(score.Note("c", 4), 1, 5)
        # add missing measures
        score.add_measures(part)
        time_pairs = [(-1, -1), (0, 0), (4, 4)]
        test_time_pairs(part, time_pairs)

    def test_times_2(self):
        # 6/8 anacrusis
        part = score.Part("id")
        # 2 divs is 1 quarter
        part.set_quarter_duration(0, 2)
        part.add(score.TimeSignature(6, 8), 0)

        # ANACRUSIS
        part.add(score.Note("c", 4), 0, 3)
        part.add(score.Measure(), 0, 3)

        part.add(score.Note("c", 4), 3, 9)

        score.add_measures(part)

        time_pairs = [(-3, -1.5), (0, 0), (6, 3)]
        test_time_pairs(part, time_pairs)
