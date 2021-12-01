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

    def test_4voice_simple(self):
        document_path = KERN_TESFILES[1]
        parts = load_kern(document_path)

    def test_1voice_simple(self):
        document_path = KERN_TESFILES[0]
        parts = load_kern(document_path)

    def test_beethoven_sonata(self):
        document_path = KERN_TESFILES[2]
        parts = load_kern(document_path)

    def test_bach_chorale(self):
        document_path = KERN_TESFILES[3]
        parts = load_kern(document_path)

    def test_bach_fugue(self):
        document_path = KERN_TESFILES[4]
        parts = load_kern(document_path)