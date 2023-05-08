#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module includes tests for display utilities. Since these utilities
rely on externally installed software (e.g., MuseScore, Lilypond),
they cannot be automatically tested by GitHub.
"""
import os
import tempfile
import unittest

import numpy as np

from partitura.utils.misc import concatenate_images, PIL_EXISTS, Image

from tests import PNG_TESTFILES


class TestMuseScoreExport(unittest.TestCase):
    def test_concat_images(self):
        """
        Test `partitura.utils.misc.concatenate_images`
        """
        if PIL_EXISTS:

            for fn in PNG_TESTFILES:
                filenames = [fn, fn]

                # original image
                oimage = Image.open(fn)

                # images concatenated vertically
                cimage_vertical = concatenate_images(
                    filenames=filenames,
                    out=None,
                    concat_mode="vertical",
                )

                tmpdir = tempfile.gettempdir()

                ofn = os.path.join(tmpdir, "test_output.png")
                concatenate_images(
                    filenames=filenames,
                    out=ofn,
                    concat_mode="vertical",
                )
                reloaded_image = Image.open(ofn)

                self.assertTrue(
                    np.allclose(
                        np.asarray(reloaded_image),
                        np.asarray(cimage_vertical),
                    )
                )

                cimage_horizontal = concatenate_images(
                    filenames=filenames,
                    out=None,
                    concat_mode="horizontal",
                )

                expected_size_vertical = (oimage.size[0], oimage.size[1] * 2)
                expected_size_horizontal = (oimage.size[0] * 2, oimage.size[1])
                self.assertTrue(cimage_vertical.size == expected_size_vertical)
                self.assertTrue(cimage_horizontal.size == expected_size_horizontal)

                oimage.close()
                reloaded_image.close()
