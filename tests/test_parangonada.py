#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Parangonada import and export.
"""

import logging
import unittest
import tempfile
import os

from partitura import load_score, load_match
from partitura.io.exportparangonada import (
    save_parangonada_alignment,
    save_alignment_for_ASAP,
    save_parangonada_csv,
)

from partitura.io.importparangonada import (
    load_parangonada_alignment,
    load_alignment_from_ASAP,
    _load_csv,
    load_parangonada_csv,
)


from tests import MOZART_VARIATION_FILES

LOGGER = logging.getLogger(__name__)

test_alignment = [
    {"label": "match", "score_id": "n01", "performance_id": "n01"},
    {"label": "insertion", "performance_id": "n02"},
    {"label": "deletion", "score_id": "n02"},
]


_performance, _alignment = load_match(filename=MOZART_VARIATION_FILES["match"])
MOZART_VARIATION_DATA = dict(
    score=load_score(MOZART_VARIATION_FILES["musicxml"]),
    performance=_performance,
    alignment=_alignment,
    parangonada_align=_load_csv(
        MOZART_VARIATION_FILES["parangonada_align"],
    ),
    parangonada_zalign=_load_csv(
        MOZART_VARIATION_FILES["parangonada_zalign"],
    ),
    parangonada_feature=_load_csv(MOZART_VARIATION_FILES["parangonada_feature"]),
    parangonada_ppart=_load_csv(MOZART_VARIATION_FILES["parangonada_ppart"]),
    parangonada_spart=_load_csv(MOZART_VARIATION_FILES["parangonada_spart"]),
)


class Ppart:
    def __init__(self):
        self.notes = [
            {
                "id": "n01",
                "track": "dummy",
                "channel": "dummy",
                "midi_pitch": "dummy",
                "note_on": "dummy",
            },
            {
                "id": "n02",
                "track": "dummy",
                "channel": "dummy",
                "midi_pitch": "dummy",
                "note_on": "dummy",
            },
        ]

    # Make dummy Ppart iterable
    def __getitem__(self, index):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return self

    def __len__(self):
        return 1


test_ppart = Ppart()


class TestIO(unittest.TestCase):
    """
    Test if the csv, tsv export and import gives the expected results.

    """

    def test_csv_import_export(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_parangonada_alignment(
                out=os.path.join(tmpdirname, "align.csv"), alignment=test_alignment
            )
            import_alignment = load_parangonada_alignment(
                os.path.join(tmpdirname, "align.csv")
            )
            equal = test_alignment == import_alignment
            self.assertTrue(equal)

    def test_tsv_import_export(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_alignment_for_ASAP(
                out=os.path.join(tmpdirname, "align.tsv"),
                performance_data=test_ppart,
                alignment=test_alignment,
            )
            import_alignment = load_alignment_from_ASAP(
                os.path.join(tmpdirname, "align.tsv")
            )
            equal = test_alignment == import_alignment
            self.assertTrue(equal)

    def test_save_parangonada_csv(self):

        with tempfile.TemporaryDirectory() as tmpdirname:

            save_parangonada_csv(
                alignment=MOZART_VARIATION_DATA["alignment"],
                performance_data=MOZART_VARIATION_DATA["performance"],
                score_data=MOZART_VARIATION_DATA["score"],
                outdir=tmpdirname,
            )

            performance, alignment, _, _ = load_parangonada_csv(tmpdirname)

            self.assertTrue(alignment == MOZART_VARIATION_DATA["alignment"])


if __name__ == "__main__":
    unittest.main()
