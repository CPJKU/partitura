"""
This file contains test functions for KERN import and export.
"""
import unittest

import partitura
from tests import KERN_TESFILES
from partitura.score import merge_parts
from partitura.io.importkern import (
    load_kern,
    parse_kern,
)

class TestImportKERN(unittest.TestCase):

    def test_example_kern(self):
        document_path = partitura.EXAMPLE_KERN
        parts = load_kern(document_path)

    def test_examples(self):
        for fn in KERN_TESFILES:
            parts = load_kern(fn)
