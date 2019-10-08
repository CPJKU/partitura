"""

This file contains test functions for Matchfile import

"""

import logging
import unittest
from tempfile import TemporaryFile

from . import MATCH_IMPORT_EXPORT_TESTFILES

from partitura.match import MatchFile


class TestLoadMatch(unittest.TestCase):

    def test_import(self):

        for fn in MATCH_IMPORT_EXPORT_TESTFILES:

            # read file
            fileData = [l.decode('utf8').strip() for l in open(fn, 'rb')]
            # parse match file
            match = MatchFile(fn)

            # Not mached lines are returned as False
            matched_lines = [True if ml else False for ml in match.lines]
            # Assert that all lines in the matchfile where matched
            self.assertTrue(sum(matched_lines), len(fileData))


if __name__ == '__main__':

    unittest.main()
