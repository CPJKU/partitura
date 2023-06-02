#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the methods for estimating metrical information.
"""
import numpy as np

import unittest
from tempfile import TemporaryFile

from tests import VOSA_TESTFILES

from partitura import load_musicxml
from partitura.musicanalysis import estimate_time
import partitura


class TestTempoMeterBeats(unittest.TestCase):
    """
    Test tempo, meter numerator, and beat estimation.
    """

    score = load_musicxml(VOSA_TESTFILES[0])
    tempometerbeats = estimate_time(score)

    def testtempo(self):

        some_performance_notes = np.array(
            [
                (5.7025, 2.4375, 40, 22, 1, 0, "n1"),
                (5.70375, 2.43625, 64, 54, 1, 0, "n2"),
                (5.77625, 2.36375, 56, 26, 1, 0, "n3"),
                (6.4325, 1.7075, 47, 20, 1, 0, "n4"),
                (6.9725, 1.1675, 63, 52, 1, 0, "n6"),
                (7.47625, 0.66375, 64, 59, 1, 0, "n8"),
                #
                (8.03375, 4.20625, 66, 58, 1, 0, "n11"),
                (8.06875, 2.04125, 35, 30, 1, 0, "n12"),
                (8.06875, 4.17125, 63, 41, 1, 0, "n13"),
                (8.09, 0.625, 57, 32, 1, 0, "n14"),
                (8.70375, 1.40625, 47, 31, 1, 0, "n15"),
                (9.2075, 0.9025, 57, 40, 1, 0, "n17"),
                (9.66625, 0.4825, 47, 30, 1, 0, "n18"),
                #
                (10.1375, 2.1025, 57, 39, 1, 0, "n20"),
                (10.14, 2.1, 35, 30, 1, 0, "n21"),
                (10.63, 1.61, 68, 57, 1, 0, "n22"),
                (11.09625, 1.14375, 68, 63, 1, 0, "n25"),
                (11.56, 0.68, 66, 65, 1, 0, "n28"),
                #
                (12.15875, 4.14125, 68, 61, 1, 0, "n31"),
                (12.18125, 1.98875, 40, 29, 1, 0, "n32"),
                (12.1875, 2.0675, 64, 48, 1, 0, "n33"),
                (12.1975, 1.9725, 56, 33, 1, 0, "n34"),
                (12.82, 1.35, 47, 27, 1, 0, "n35"),
                (13.30625, 0.86375, 56, 36, 1, 0, "n37"),
                (13.8325, 0.47125, 47, 25, 1, 0, "n38"),
            ],
            dtype=[
                ("onset_sec", "<f4"),
                ("duration_sec", "<f4"),
                ("pitch", "<i4"),
                ("velocity", "<i4"),
                ("track", "<i4"),
                ("channel", "<i4"),
                ("id", "<U256"),
            ],
        )
        result = estimate_time(some_performance_notes)

        self.assertTrue(np.all(result["meter_numerator"] == 4), "Incorrect meter.")
        self.assertTrue(len(result["beats"]) == 16, "Incorrect number of beats.")
        self.assertTrue(np.isclose(result["tempo"], 111, atol=1.0), "Incorrect tempo.")
