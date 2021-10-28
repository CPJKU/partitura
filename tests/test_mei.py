"""
This file contains test functions for MEI export
"""

import unittest

from tests import MEI_TESTFILES
from partitura import load_musicxml, load_mei


class TestSaveMEI(unittest.TestCase):

    # def test_save_mei(self):

    #     with open(EXAMPLE_MEI, 'r') as f:
    #         target_mei = f.read()

    #     mei = save_mei(load_musicxml(EXAMPLE_MUSICXML), title_text='score_example')
    #     msg = "Export of MEI of file {} does not yield identical result".format(EXAMPLE_MEI)

    #     self.assertTrue(mei.decode('utf-8') == target_mei, msg)

    def test_import_mei(self):
        part = load_mei(MEI_TESTFILES[4])
        self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()

