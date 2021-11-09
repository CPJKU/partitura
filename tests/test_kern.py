"""
This file contains test functions for MEI export
"""
import unittest

from tests import KERN_TESFILES
from partitura import load_musicxml, load_kern
from partitura.io.importkern import (
    load_kern,
    _parse_kern,
)



class TestImportMEI(unittest.TestCase):
    def test_main_part_group1(self):
        document_path = KERN_TESFILES[1]
        print(_parse_kern(document_path))
        load_kern(KERN_TESFILES[1])
