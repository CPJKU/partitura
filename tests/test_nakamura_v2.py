"""

This file contains test functions for the import of Nakamura et al.'s match and corresp file formats.

"""

import unittest
from partitura.io.importnakamura_v2 import (
    load_nakamuracorresp_v2,
)

from . import NAKAMURA_IMPORT_TESTFILES

# from partitura import load_nakamuracorresp, load_nakamuramatch


class TestLoadNakamura(unittest.TestCase):
    def test_import(self):
        for fn in NAKAMURA_IMPORT_TESTFILES:
            if "corresp" in fn:
                performance, score, align = load_nakamuracorresp_v2(fn)
                performance_ids = set(
                    item["performance_id"]
                    for item in align
                    if item["label"] in ("match", "insertion")
                )
                score_ids = set(
                    item["score_id"]
                    for item in align
                    if item["label"] in ("match", "deletion")
                )
                self.assertTrue(len(performance_ids) == len(performance))
                self.assertTrue(len(score_ids) == len(score))
            else:
                # TODO: test match
                pass


if __name__ == "__main__":

    unittest.main()
