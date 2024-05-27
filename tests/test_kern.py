#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for KERN import and export.
"""
import unittest

import partitura
import os
from tests import KERN_TESTFILES, KERN_TIES, KERN_PATH
from tempfile import TemporaryDirectory
from partitura.score import merge_parts
from partitura.utils import ensure_notearray
from partitura.io.importkern import load_kern
from partitura.io.exportkern import save_kern
from partitura import load_musicxml
import numpy as np


class TestImportKERN(unittest.TestCase):
    def test_example_kern(self):
        document_path = partitura.EXAMPLE_KERN
        kern_part = merge_parts(load_kern(document_path))
        xml_part = load_musicxml(partitura.EXAMPLE_MUSICXML)
        ka = ensure_notearray(kern_part)
        xa = ensure_notearray(xml_part)
        self.assertTrue(
            np.all(ka["onset_beat"] == xa["onset_beat"]),
            "Kern onset beats do not match target",
        )
        self.assertTrue(
            np.all(ka["duration_beat"] == xa["duration_beat"]),
            "Kern duration beats do not match target.",
        )

    def test_examples(self):
        for fn in KERN_TESTFILES:
            score = load_kern(fn)
            self.assertTrue(True)

    def test_chorale_import(self):
        file_path = os.path.join(KERN_PATH, "chor228.krn")
        score = load_kern(file_path)
        num_measures = 8
        num_parts = 4
        num_notes = 102
        self.assertTrue(len(score.parts) == num_parts)
        self.assertTrue(all([len(part.measures) == num_measures for part in score.parts]))
        self.assertTrue(len(score.note_array()) == num_notes)

    def test_tie_mismatch(self):

        fn = KERN_TIES[0]
        score = load_kern(fn)

        self.assertTrue(True)

    def test_spline_splitting(self):
        file_path = os.path.join(KERN_PATH, "spline_splitting.krn")
        score = load_kern(file_path)
        num_parts = 4
        voices_per_part = [2, 1, 1, 2]
        self.assertTrue(num_parts == len(score.parts))
        for i, part in enumerate(score.parts):
            vn = part.note_array()["voice"].max()
            self.assertTrue(voices_per_part[i] == vn)

    def test_import_export(self):
        imported_score = load_kern(partitura.EXAMPLE_KERN)
        with TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test.match")
            save_kern(imported_score, out)
            exported_score = load_kern(out)
        im_na = imported_score.note_array(include_staff=True)
        ex_na = exported_score.note_array(include_staff=True)
        self.assertTrue(np.all(im_na["onset_beat"] == ex_na["onset_beat"]))
        self.assertTrue(np.all(im_na["duration_beat"] == ex_na["duration_beat"]))
        self.assertTrue(np.all(im_na["pitch"] == ex_na["pitch"]))
        self.assertTrue(np.all(im_na["staff"] == ex_na["staff"]))
        # NOTE: Voices are not the same because of the way voices are assigned in merge_parts


# if __name__ == "__main__":
#     unittest.main()
