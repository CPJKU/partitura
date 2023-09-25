#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the methods computing tonal tension.
"""
import unittest

from partitura import (
    EXAMPLE_MIDI,
    EXAMPLE_MUSICXML,
    load_performance_midi,
    load_musicxml,
)

from partitura.musicanalysis.tonal_tension import (
    prepare_note_array,
    estimate_tonaltension,
)
import numpy as np


class TestTonalTension(unittest.TestCase):
    score = load_musicxml(EXAMPLE_MUSICXML)
    performance = load_performance_midi(EXAMPLE_MIDI)[0]

    def test_prepare_notearray(self):
        target_note_array = np.array(
            [
                (0.0, 2.0, 69, 64, "n0", "A", 0, 4, 0, -1),
                (1.0, 1.0, 72, 64, "n1", "C", 0, 5, 0, -1),
                (1.0, 1.0, 76, 64, "n2", "E", 0, 5, 0, -1),
            ],
            dtype=[
                ("onset_sec", "<f4"),
                ("duration_sec", "<f4"),
                ("pitch", "<i4"),
                ("velocity", "<i4"),
                ("id", "<U256"),
                ("step", "<U1"),
                ("alter", "<i8"),
                ("octave", "<i8"),
                ("ks_fifths", "<i4"),
                ("ks_mode", "<i4"),
            ],
        )
        note_array = prepare_note_array(self.performance)

        for name in target_note_array.dtype.names:

            self.assertTrue(
                np.all(note_array[name] == target_note_array[name]),
                f"Note arrays are not equal for field {name}",
            )

    def test_estimate_tonaltension(self):
        tonal_tension = estimate_tonaltension(self.score)

        target_tension = np.array(
            [(0, 0, 0, 0.19651566), (2.0, 0.33333334, 0.07754743, 0.13506594)],
            dtype=[
                ("onset_beat", "<f4"),
                ("cloud_diameter", "<f4"),
                ("cloud_momentum", "<f4"),
                ("tensile_strain", "<f4"),
            ],
        )
        self.assertTrue(
            np.all(tonal_tension == target_tension), "estimated tension is incorrect!"
        )
