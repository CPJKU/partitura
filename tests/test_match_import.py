#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Matchfile import
"""
import unittest
import numpy as np

from tests import MATCH_IMPORT_EXPORT_TESTFILES, MOZART_VARIATION_FILES

from partitura.io.importmatch import MatchFile, parse_matchline
from partitura import load_match, load_score, load_performance


class TestLoadMatch(unittest.TestCase):
    def test_import(self):

        for fn in MATCH_IMPORT_EXPORT_TESTFILES:

            # read file
            with open(fn) as f:

                file_contents = [parse_matchline(l) for l in f.read().splitlines()]

            # parse match file
            match = MatchFile(fn)

            # Not mached lines are returned as False
            # matched_lines = [True if ml else False for ml in match.lines]
            matched_lines = [1 for ml in match.lines if ml]
            # Assert that all lines in the matchfile where matched
            self.assertTrue(len(matched_lines), len(file_contents))

    def test_match_lines(self):

        snote_line = "snote(1-1,[E,n],4,0:1,0,1/4,-1.0,0.0,[staff1])"
        note_line = "note(0,[E,n],4,471720,472397,472397,49)"
        old_note_line = "note(0,[E,n],4,471720,472397,49)"
        snote_note_line = "snote(1-1,[E,n],4,0:1,0,1/4,-1.0,0.0,[staff1])-note(0,[E,n],4,471720,472397,472397,49)."
        snote_deletion_line = "snote(1-1,[E,n],4,0:1,0,1/4,-1.0,0.0,[staff1])-deletion."
        note_insertion_line = "insertion-" + note_line + "."
        info_line = "info(matchFileVersion,4.0)."
        meta_line = "meta(keySignature,C Maj/A min,0,-1.0)."
        sustain_line = "sustain(779,59)."
        trill_line = "trill(726-1)-note(751,[D,n],5,57357,57533,57533,60)."
        ornament_line = "ornament(726-1)-note(751,[D,n],5,57357,57533,57533,60)."

        matchlines = [
            snote_note_line,
            snote_deletion_line,
            note_insertion_line,
            info_line,
            meta_line,
            sustain_line,
            trill_line,
            ornament_line,
        ]

        for ml in matchlines:
            mo = parse_matchline(ml)
            self.assertTrue(mo.matchline, ml)

    def test_load_match(self):

        perf_match, alignment, score_match = load_match(
            filename=MOZART_VARIATION_FILES["match"],
            create_score=True,
            first_note_at_zero=True,
        )

        pna_match = perf_match.note_array()
        sna_match = score_match.note_array()

        perf_midi = load_performance(
            filename=MOZART_VARIATION_FILES["midi"],
            first_note_at_zero=True,
        )

        pna_midi = perf_midi.note_array()
        score_musicxml = load_score(
            filename=MOZART_VARIATION_FILES["musicxml"],
        )

        sna_musicxml = score_musicxml.note_array()

        for note in alignment:

            # check score info in match and MusicXML
            if "score_id" in note:

                idx_smatch = np.where(sna_match["id"] == note["score_id"])[0]
                idx_sxml = np.where(sna_musicxml["id"] == note["score_id"])[0]

                self.assertTrue(
                    sna_match[idx_smatch]["pitch"] == sna_musicxml[idx_sxml]["pitch"]
                )

                self.assertTrue(
                    np.isclose(
                        sna_match[idx_smatch]["onset_beat"],
                        sna_match[idx_sxml]["onset_beat"],
                    )
                )

                self.assertTrue(
                    np.isclose(
                        sna_match[idx_smatch]["duration_beat"],
                        sna_match[idx_sxml]["duration_beat"],
                    )
                )

            # check performance info in match and MIDI
            if "performance_id" in note:

                idx_pmatch = np.where(pna_match["id"] == note["performance_id"])[0]
                idx_pmidi = np.where(pna_midi["id"] == note["performance_id"])[0]

                self.assertTrue(
                    pna_match[idx_pmatch]["pitch"] == pna_midi[idx_pmidi]["pitch"]
                )

                self.assertTrue(
                    np.isclose(
                        pna_match[idx_pmatch]["onset_sec"],
                        pna_match[idx_pmidi]["onset_sec"],
                    )
                )

                self.assertTrue(
                    np.isclose(
                        pna_match[idx_pmatch]["duration_sec"],
                        pna_match[idx_pmidi]["duration_sec"],
                    )
                )

                

if __name__ == "__main__":

    unittest.main()
