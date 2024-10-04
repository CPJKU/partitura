#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for clef related methods.
"""
import unittest
from tests import (
    CLEF_TESTFILES,
)
from partitura import load_musicxml
from partitura.musicanalysis import compute_note_array
