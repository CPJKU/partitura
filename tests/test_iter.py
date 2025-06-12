"""
This module tests that iter_all can be called with both an individual class and
a list/tuple of classes.
"""

import unittest
from tests import MUSICXML_SCORE_OBJECT_TESTFILES
import partitura as prt
from collections import defaultdict


class TestingIterMethods(unittest.TestCase):
    """
    A test class for testing the iter_all method in the Partitura library.

    This test suite checks various behaviors of the iter_all method, including
    its ability to iterate over different classes and handle various method
    parameters.
    """

    def setUp(self):
        """
        Prepare test data by loading MusicXML files for testing.

        Loads each MusicXML file from MUSICXML_SCORE_OBJECT_TESTFILES into
        the self.scores list, ensuring note IDs are preserved.

        Raises:
            AssertionError: If no test files are found in MUSICXML_SCORE_OBJECT_TESTFILES.
        """
        self.scores = []
        assert len(MUSICXML_SCORE_OBJECT_TESTFILES) > 0, (
            "no MUSICXML_SCORE_OBJECT_TESTFILES found"
        )
        for fn in MUSICXML_SCORE_OBJECT_TESTFILES:
            score = prt.load_musicxml(fn, force_note_ids="keep")
            assert len(score.parts) > 0, (
                f"No parts found in {MUSICXML_SCORE_OBJECT_TESTFILES[i]}"
            )
            self.scores.append(score)

    def _test_iter_methods(self, kwargs):
        """
        Test the iter_all method with various class iteration scenarios.

        This method tests iter_all with different input configurations:
        1. Single class iteration
        2. Multiple class iteration (as tuple)
        3. Multiple class iteration (as list)

        Args:
            kwargs (dict): Optional keyword arguments to pass to iter_all method.
                           Can include 'include_subclasses' and 'mode'.

        Raises:
            AssertionError: If the sum of iterated items for individual classes
            not does not match the number of iterated items for the tuple/list of
            classes.
        """
        for i, score in enumerate(self.scores):
            part = score.parts[0]

            # Define two different classes to test iteration
            cls1 = prt.score.Note
            cls2 = prt.score.TimeSignature

            # Iterate through all notes in the part
            items1 = tuple(part.iter_all(cls1, **kwargs))
            # Iterate through all time signatures in the part
            items2 = tuple(part.iter_all(cls2, **kwargs))
            # Count the number of items for each class
            n1 = len(items1)
            n2 = len(items2)

            # Create a tuple of classes to iterate through
            cls3 = tuple((cls1, cls2))
            # Iterate through both notes and time signatures simultaneously
            items3 = tuple(part.iter_all(cls3, **kwargs))
            # Check that the total number of items matches the sum of individual class items
            n3 = len(items3)
            assert n3 == n1 + n2

            # Repeat the same test using a list of classes instead of a tuple
            cls4 = list((cls1, cls2))
            items4 = tuple(part.iter_all(cls4, **kwargs))
            n4 = len(items4)
            # Verify that the number of items is consistent
            assert n4 == n1 + n2

    def test_iter_warning(self):
        """ """
        for i, score in enumerate(self.scores):
            part = score.parts[0]

            include_subclasses = True
            cls1 = [prt.score.Note, prt.score.GenericNote]

            # assert we get a warning because Note is a GenericNote subclass
            with self.assertWarns(UserWarning):
                items1 = tuple(
                    part.iter_all(cls1, include_subclasses=include_subclasses)
                )

            cls2 = prt.score.GenericNote
            items2 = tuple(part.iter_all(cls2, include_subclasses=include_subclasses))

            assert len(items1) == len(items2)

    def test_iter_methods(self):
        """
        Test the iter_all method with various parameter combinations.

        This method tests iter_all with:
        1. Default parameters
        2. Combinations of include_subclasses (True/False) and modes: "starting"
           and "ending"
        """
        kwargs = {}
        self._test_iter_methods(kwargs)

        for incl_subcl in (True, False):
            for mode in ("starting", "ending"):
                kwargs = {"include_subclasses": incl_subcl, "mode": mode}
                self._test_iter_methods(kwargs)

    def test_iter_order(self):
        """Check that the order of objects follows the order of target classes"""

        def classes_by_time(items):
            result = defaultdict(list)
            for item in items:
                result[item.start.t].append(type(item))
            return result

        for i, score in enumerate(self.scores):
            part = score.parts[0]

            cls1 = [prt.score.Note, prt.score.Measure]
            items1 = tuple(part.iter_all(cls1))

            result = classes_by_time(items1)

            for classes in result.values():
                if prt.score.Note in classes and prt.score.Measure in classes:
                    # check that Note goes before Measure
                    assert classes.index(prt.score.Note) < classes.index(
                        prt.score.Measure
                    )

            assert isinstance(items1[0], prt.score.Note)
            assert isinstance(items1[1], prt.score.Measure)

            cls2 = [prt.score.Measure, prt.score.Note]
            items2 = tuple(part.iter_all(cls2))
            assert len(items1) == len(items2)
            assert items1 != items2

            result = classes_by_time(items2)
            for classes in result.values():
                if prt.score.Note in classes and prt.score.Measure in classes:
                    # check that Measure goes before Note
                    assert classes.index(prt.score.Measure) < classes.index(
                        prt.score.Note
                    )
