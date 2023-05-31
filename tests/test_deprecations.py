#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes tests for deprecation utilities.
"""
import unittest
import warnings
import numpy as np

from partitura.utils import deprecated_alias, deprecated_parameter

RNG = np.random.RandomState(1984)


class TestDeprecations(unittest.TestCase):
    @deprecated_alias(old_p1="new_p1", old_p2="new_p2")
    def func_alias(self, new_p1=None, new_p2=None, **kwargs):

        crit = not ("old_p1" in kwargs or "old_p2" in kwargs)

        self.assertTrue(crit)
        self.assertTrue(new_p1 is not None)
        self.assertTrue(new_p2 is not None)

    @deprecated_parameter(*[f"deprecated{i}" for i in range(10)])
    def func_parameter(self, new_p1=None, new_p2=None, **kwargs):

        crit = not any([f"deprecated{i}" in kwargs for i in range(10)])

        self.assertTrue(crit)
        self.assertTrue(new_p1 is not None)
        self.assertTrue(new_p2 is not None)

    def test_deprecated_alias(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for old_p1, old_p2 in RNG.rand(10, 2):
                self.func_alias(old_p1=old_p1, old_p2=old_p2)

    def test_deprecated_parameter(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for rp in RNG.rand(10, 12):
                kwargs = dict(
                    [("new_p1", rp[0]), ("new_p2", rp[1])]
                    + [(f"deprecated{i}", rp[i + 2]) for i in range(10)]
                )
                self.func_parameter(**kwargs)
