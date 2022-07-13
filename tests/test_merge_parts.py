#!/usr/bin/env python
import numpy as np
import logging
import unittest
from pathlib import Path

from partitura import load_musicxml
from partitura.score import merge_parts, Part, iter_parts
from partitura.utils.music import ensure_notearray

from tests import MERGE_PARTS_TESTFILES

LOGGER = logging.getLogger(__name__)


class TestMergeParts(unittest.TestCase):
    """
    Test merge parts utility
    """

    def test_list_of_parts_and_partgroup(self):
        parts = load_musicxml(MERGE_PARTS_TESTFILES[1])
        merged_part = merge_parts(parts)
        note_array = merged_part.note_array()
        expected_onsets = [0, 0, 0, 0, 0, 0, 12, 15, 24, 24, 32, 40, 48]
        expected_pitches = [48, 50, 53, 62, 67, 69, 64, 65, 47, 69, 67, 64, 60]
        expected_duration = [24, 24, 24, 12, 24, 24, 3, 3, 24, 8, 8, 8, 24]
        self.assertTrue(np.array_equal(note_array["onset_div"], expected_onsets))
        self.assertTrue(np.array_equal(note_array["pitch"], expected_pitches))
        self.assertTrue(np.array_equal(note_array["duration_div"], expected_duration))

    def test_different_divs(self):
        parts = load_musicxml(MERGE_PARTS_TESTFILES[2])
        merged_part = merge_parts(parts)
        note_array = merged_part.note_array()
        expected_onsets = [0, 0, 0, 0, 0, 0, 12, 15, 24, 24, 32, 40, 48]
        expected_pitches = [48, 50, 53, 62, 67, 69, 64, 65, 47, 69, 67, 64, 60]
        self.assertTrue(np.array_equal(note_array["onset_div"], expected_onsets))
        self.assertTrue(np.array_equal(note_array["pitch"], expected_pitches))

    def test_compare_normal_and_different_divs(self):
        parts_normal = load_musicxml(MERGE_PARTS_TESTFILES[1])
        parts_diff = load_musicxml(MERGE_PARTS_TESTFILES[2])
        merged_part_normal = merge_parts(parts_normal)
        merged_part_diff = merge_parts(parts_diff)
        note_array_normal = merged_part_normal.note_array()
        note_array_diff = merged_part_diff.note_array()
        self.assertTrue(
            np.array_equal(
                note_array_normal["onset_beat"], note_array_diff["onset_beat"]
            )
        )

    def test_merge_single_part(self):
        parts = load_musicxml(MERGE_PARTS_TESTFILES[3])
        merged_part = merge_parts(parts)
        self.assertTrue(merged_part == parts)

    # def test_merge_interpolation(self):
    #     parts = load_musicxml(MERGE_PARTS_TESTFILES[4])
    #     merged_part = merge_parts(parts)
    #     self.assertTrue(isinstance(merged_part, Part))

    def test_merge_voices(self):
        parts = load_musicxml(MERGE_PARTS_TESTFILES[6])
        merged_part = merge_parts(parts)
        note_array = merged_part.note_array()
        expected_voices = [3,2,1,1]
        self.assertTrue(note_array["voice"].tolist() == expected_voices)