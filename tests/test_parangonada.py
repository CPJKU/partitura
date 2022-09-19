"""

This file contains test functions for MusicXML import and export.

"""

import logging
import unittest
import tempfile
import os

from partitura.io.exportparangonada import (
    save_alignment_for_parangonada,
    load_alignment_from_parangonada,
    save_alignment_for_ASAP,
    load_alignment_from_ASAP,
)

LOGGER = logging.getLogger(__name__)

test_alignment = [
    {"label": "match", "score_id": "n01", "performance_id": "n01"},
    {"label": "insertion", "performance_id": "n02"},
    {"label": "deletion", "score_id": "n02"},
]


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


test_ppart = Ppart()


class TestIO(unittest.TestCase):
    """
    Test if the csv, tsv export and import gives the expected results.

    """

    def test_csv_import_export(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_alignment_for_parangonada(
                os.path.join(tmpdirname, "align.csv"), test_alignment
            )
            import_alignment = load_alignment_from_parangonada(
                os.path.join(tmpdirname, "align.csv")
            )
            equal = test_alignment == import_alignment
            self.assertTrue(equal)

    def test_tsv_import_export(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_alignment_for_ASAP(
                os.path.join(tmpdirname, "align.tsv"), test_ppart, test_alignment
            )
            import_alignment = load_alignment_from_ASAP(
                os.path.join(tmpdirname, "align.tsv")
            )
            equal = test_alignment == import_alignment
            self.assertTrue(equal)


if __name__ == "__main__":
    unittest.main()
