"""

This file contains test functions for the import of Nakamura et al.'s match and corresp file formats.

"""

import logging
import unittest
from tempfile import TemporaryFile

from . import NAKAMURA_IMPORT_TESTFILES

#from partitura.io.importnakamura import ?
from partitura import load_nakamuracorresp, load_nakamuramatch


class TestLoadNakamura(unittest.TestCase):

    def test_import(self):

        for fn in NAKAMURA_IMPORT_TESTFILES:

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

        snote_line = 'snote(1-1,[E,n],4,0:1,0,1/4,-1.0,0.0,[staff1])'


        self.assertTrue(mo.matchline, ml)


if __name__ == '__main__':

    unittest.main()
