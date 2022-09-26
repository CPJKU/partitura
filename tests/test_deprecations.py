import unittest
import numpy as np

from partitura.utils import deprecated_alias

RNG = np.random.RandomState(1984)


class TestDeprecations(unittest.TestCase):
    @deprecated_alias(old_p1="new_p1", old_p2="new_p2")
    def func(self, new_p1=None, new_p2=None, **kwargs):

        crit = "old_p1" in kwargs or "old_p2" in kwargs

        self.assertTrue(not crit)
        self.assertTrue(new_p1 is not None)
        self.assertTrue(new_p2 is not None)

    def test_deprecated_alias(self):

        for old_p1, old_p2 in RNG.rand(10, 2):
            self.func(old_p1=old_p1, old_p2=old_p2)
