"""

This file contains test functions for the import of Nakamura et al.'s match and corresp file formats.

"""

import logging
import unittest
from tempfile import TemporaryFile
from partitura.io.importnakamura import parse_nakamuracorrespline  

from . import NAKAMURA_IMPORT_TESTFILES

#from partitura.io.importnakamura import ?
from partitura import load_nakamuracorresp, load_nakamuramatch


class TestLoadNakamura(unittest.TestCase):

    def test_import(self):
        for fn in NAKAMURA_IMPORT_TESTFILES:
            if "corresp" in fn:
                # read file
                with open(fn) as f:

                    file_contents = [(l,parse_nakamuracorrespline(l)) for l in f.read().splitlines()]


                for fc in file_contents:
                # assert that the lines are correctly encoded
                    self.assertTrue(fc[0], fc[1].corresp_line)

if __name__ == '__main__':

    unittest.main()
