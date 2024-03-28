#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the fluidsynth methods.
"""
import unittest

import numpy as np
from scipy.io import wavfile
import tempfile

from partitura.utils.fluidsynth import (
    synthesize_fluidsynth,
    HAS_FLUIDSYNTH,
    SAMPLE_RATE,
)

from partitura import EXAMPLE_MUSICXML, load_score, load_performance_midi

from partitura.io.exportaudio import save_wav_fluidsynth

from tests import MOZART_VARIATION_FILES

RNG = np.random.RandomState(1984)

if HAS_FLUIDSYNTH:

    class TestSynthesize(unittest.TestCase):

        score = load_score(EXAMPLE_MUSICXML)

        def test_synthesize(self):

            score_na = self.score.note_array()

            duration_beats = (
                score_na["onset_beat"] + score_na["duration_beat"]
            ).max() - score_na["onset_beat"].min()

            for bpm in RNG.randint(30, 200, size=10):

                for samplerate in [12000, 16000, 22000, SAMPLE_RATE]:

                    duration_sec = duration_beats * 60 / bpm
                    y = synthesize_fluidsynth(
                        note_info=self.score,
                        samplerate=samplerate,
                        bpm=bpm,
                    )

                    expected_length = np.round(duration_sec * samplerate)

                    self.assertTrue(len(y) == expected_length)

                    self.assertTrue(isinstance(y, np.ndarray))

    class TestSynthExport(unittest.TestCase):

        test_files = [
            load_score(MOZART_VARIATION_FILES["musicxml"]),
            load_performance_midi(MOZART_VARIATION_FILES["midi"]),
        ]

        def export(self, note_info):

            y = synthesize_fluidsynth(
                note_info=note_info,
                samplerate=SAMPLE_RATE,
                bpm=60,
            )

            with tempfile.TemporaryFile(suffix=".wav") as filename:

                save_wav_fluidsynth(
                    input_data=note_info,
                    out=filename,
                    samplerate=SAMPLE_RATE,
                    bpm=60,
                )

                sr_rec, rec_audio = wavfile.read(filename)

                self.assertTrue(sr_rec == SAMPLE_RATE)
                self.assertTrue(len(rec_audio) == len(y))
                self.assertTrue(
                    np.allclose(
                        rec_audio / rec_audio.max(),
                        y / y.max(),
                        atol=1e-4,
                    )
                )

        def test_export(self):

            for note_info in self.test_files:

                self.export(note_info)
