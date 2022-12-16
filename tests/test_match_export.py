#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Matchfile import
"""
import unittest
import numpy as np
import re
import os
from tempfile import TemporaryDirectory

from tests import MOZART_VARIATION_FILES

from partitura.io.exportmatch import matchfile_from_alignment, save_match
from partitura.io.importmatch import load_match
from partitura import load_score


class TestExportMatch(unittest.TestCase):
    def test_matchfile_from_alignment(self):
        """
        test `matchfile_from_alignment`
        """
        score_fn = MOZART_VARIATION_FILES["musicxml"]

        score = load_score(score_fn)
        spart = score[0]
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, alignment = load_match(match_fn)

        matchfile = matchfile_from_alignment(
            alignment=alignment,
            ppart=performance[0],
            spart=spart,
            assume_part_unfolded=True,
        )

        sna = spart.note_array()
        pna = performance.note_array()

        # assert that matchfile contains the same number of notes
        self.assertTrue(len(sna) == len(matchfile.snotes))
        self.assertTrue(len(pna) == len(matchfile.notes))

        snote_ids = [n.Anchor for n in matchfile.snotes]
        pnote_ids = [n.Id for n in matchfile.notes]
        # assert that all snotes in the matchfile are in the note array
        self.assertTrue(all([n.Anchor in sna["id"] for n in matchfile.snotes]))

        # assert that all notes in the score are in the matchfile
        self.assertTrue(all([nid in snote_ids for nid in sna["id"]]))

        # assert that all notes in the matchfile are in the note array
        self.assertTrue(all([n.Id in pna["id"] for n in matchfile.notes]))

        # assert that all notes in the performance are in the matchfile
        self.assertTrue(all(nid in pnote_ids for nid in pna["id"]))

        for ml in matchfile.lines:
            self.assertTrue(isinstance(ml.matchline, str))

    def test_save_match(self):
        """
        Test save_match
        """
        score_fn = MOZART_VARIATION_FILES["musicxml"]

        score = load_score(score_fn)
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, alignment = load_match(match_fn)
        pna1 = performance.note_array()
        with TemporaryDirectory() as tmpdir:

            out = os.path.join(tmpdir, "test.match")
            save_match(
                alignment=alignment,
                performance_data=performance,
                score_data=score,
                out=out,
                performer="A Pianist",
                composer="W. A. Mozart",
                piece="mozart_k265_var1",
                score_filename=os.path.basename(score_fn),
                performance_filename=os.path.basename(MOZART_VARIATION_FILES["midi"]),
                assume_unfolded=True,
            )

            perf_from_saved_match, alignment_from_saved_match = load_match(out)

        pna2 = perf_from_saved_match.note_array()

        # Test that performance data is the same
        for field in (
            "onset_sec",
            "duration_sec",
            "onset_tick",
            "duration_tick",
            "pitch",
            "velocity",
        ):
            self.assertTrue(np.allclose(pna2[field], pna1[field]))

        # Test that the alignment info is correct
        for al in alignment_from_saved_match:
            self.assertTrue(al in alignment)

        for al in alignment:
            self.assertTrue(al in alignment_from_saved_match)
