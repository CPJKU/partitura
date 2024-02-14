#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for KERN import and export.
"""
import unittest

import partitura
import os
from tests import KERN_TESTFILES, KERN_TIES, KERN_PATH
from partitura.score import merge_parts
from partitura.utils import ensure_notearray
from partitura.io.importkern_v2 import load_kern
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
        exported_score = save_kern(imported_score)
        x = np.loadtxt(partitura.EXAMPLE_KERN, comments="!!", dtype=str, encoding="utf-8", delimiter="\t")
        self.assertTrue(np.all(x == exported_score))


# if __name__ == "__main__":
#     unittest.main()
