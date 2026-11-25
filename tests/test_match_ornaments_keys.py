#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for ornaments and key signatures in match files
"""
import os
import unittest

from tempfile import TemporaryDirectory

from partitura.io.exportmatch import save_match
from partitura.io.importmatch import load_match, load_matchfile


ORNAMENT_LINES = [
    "info(matchFileVersion,1.0.0).",
    "info(piece,-).",
    "info(midiClockUnits,480).",
    "info(midiClockRate,500000).",
    "scoreprop(keySignature,C,1:1,0,0.0000).",
    "scoreprop(timeSignature,4/4,1:1,0,0.0000).",
    "snote(n1,[C,n],4,1:1,0,1/4,0.0000,1.0000,[v1,staff1])-note(n1,60,0,100,64,0,0).",
    "ornament(n1,[trill])-note(n2,62,100,200,64,0,0).",
    "ornament(n1,[trill,mordent])-note(n3,60,200,300,64,0,0).",
    "snote(n2,[D,n],4,1:2,0,1/4,1.0000,2.0000,[v1,staff1])-note(n4,62,480,900,64,0,0).",
]


class TestOrnaments(unittest.TestCase):
    """
    Test loading and saving ornaments
    """

    def test_load_ornaments(self):
        """
        Notes in ornament lines are part of the performance
        """
        with TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, "test.match")
            with open(fn, "w") as f:
                f.write("\n".join(ORNAMENT_LINES))
            performance, alignment, scr = load_match(fn, create_score=True)

            mf = load_matchfile(fn)
            self.assertEqual([n.Id for n in mf.notes], ["n1", "n2", "n3", "n4"])
            self.assertEqual([n.Id for n in mf.iter_notes()], [n.Id for n in mf.notes])

            pna = performance.note_array()
            self.assertEqual(sorted(pna["id"]), ["n1", "n2", "n3", "n4"])
            self.assertEqual(
                [(a["performance_id"], a["type"]) for a in alignment if "type" in a],
                [("n2", ["trill"]), ("n3", ["trill", "mordent"])],
            )
            # ornaments are not score notes
            self.assertEqual(list(scr.note_array()["id"]), ["n1", "n2"])

            # the alignment can be saved
            out = os.path.join(tmpdir, "out.match")
            save_match(alignment, performance, scr, out=out)
            performance2, alignment2 = load_match(out)
            self.assertEqual(alignment2, alignment)
            self.assertEqual(len(performance2.note_array()), 4)

            # the type of the ornament can also be a string
            for al in alignment:
                if al["label"] == "ornament":
                    al["type"] = "trill"
            save_match(alignment, performance, scr, out=out)
            _, alignment2 = load_match(out)
            self.assertEqual(
                [a["type"] for a in alignment2 if "type" in a], [["trill"], ["trill"]]
            )


if __name__ == "__main__":
    unittest.main()
