#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the voice estimation methods.
"""
import numpy as np

import unittest
from tempfile import TemporaryFile

from tests import VOSA_TESTFILES

from partitura import load_musicxml
from partitura.musicanalysis import estimate_voices
import partitura


class TestVoSA(unittest.TestCase):
    """
    Test VoSA
    """

    score = load_musicxml(VOSA_TESTFILES[0])

    def test_vosa_chew(self):
        # Example from Chew and Wu.
        voices = estimate_voices(self.score, monophonic_voices=True)
        # ground_truth_voices = np.array([3, 2, 1, 3, 3, 2, 3, 3, 2, 3,
        #                                 3, 1, 1, 2, 1, 1, 3, 2, 1, 1])
        # ground_truth_voices = np.array(
        #     [1, 2, 3, 1, 1, 2, 1, 1, 2, 1, 1, 3, 3, 2, 3, 3, 1, 2, 3, 3]
        # )
        ground_truth_voices = np.array(
            [3, 2, 1, 1, 2, 1, 1, 2, 1, 1, 3, 1, 3, 3, 2, 3, 3, 2, 1, 3]
        )
        self.assertTrue(
            np.all(voices == ground_truth_voices), "Incorrect voice assignment."
        )

        self.assertTrue(True)

    def test_vosa_chew_chordnotes(self):
        # Example from Chew and Wu.
        voices = estimate_voices(self.score, monophonic_voices=False)
        print(voices)
        # ground_truth_voices = np.array(
        #     [1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 3, 2, 2, 1, 1, 2, 2]
        # )
        ground_truth_voices = np.array(
            [3, 3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 3, 3, 1, 3, 3, 2, 2, 3]
        )
        self.assertTrue(
            np.all(voices == ground_truth_voices), "Incorrect voice assignment."
        )
        self.assertTrue(True)
