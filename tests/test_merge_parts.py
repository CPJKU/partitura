#!/usr/bin/env python
import numpy as np
import logging
import unittest

from partitura.utils.music import compute_pianoroll, pianoroll_to_notearray

LOGGER = logging.getLogger(__name__)


class TestMergeParts(unittest.TestCase):
    """
    Test merge parts utility

    Asap failing in 
    [('data/raw/asap-dataset/Chopin/Berceuse_op_57/xml_score.musicxml',
  partitura.score.PartGroup),
 ('data/raw/asap-dataset/Liszt/Hungarian_Rhapsodies/6/xml_score.musicxml',
  partitura.score.PartGroup),
 ('data/raw/asap-dataset/Ravel/Gaspard_de_la_Nuit/1_Ondine/xml_score.musicxml',
  list),
 ('data/raw/asap-dataset/Ravel/Miroirs/3_Une_Barque/xml_score.musicxml',
  partitura.score.PartGroup),
 ('data/raw/asap-dataset/Scriabin/Sonatas/5/xml_score.musicxml',
  partitura.score.PartGroup)]
    """

