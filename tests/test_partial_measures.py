#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for partial measures.
"""
import unittest

import os
import numpy as np
import partitura
from partitura import load_musicxml
from partitura import score

import xml.etree.ElementTree as ET

# TODO tmp, del later
import warnings
warnings.filterwarnings("ignore")


 # TODO # import error, doesn't recognise path
# from tests import MUSICXML_PARTIAL_MEASURES_TESTFILES 
tests_dir = os.path.dirname(os.path.abspath(__file__))
tests = os.path.dirname(os.path.abspath(__file__))
partial_measures_test_file = os.path.join(tests, 'data', 'musicxml', 'test_partial_measures.musicxml')
consecutive_partial_measures_test_file = os.path.join(tests, 'data', 'musicxml', 'test_partial_measures_consecutive.musicxml')

# test_partial_measures.musicxml: Var. V from Sonata K331, 1. mov
spart = load_musicxml(partial_measures_test_file).parts[0]
snote_array = spart.note_array()
# print(type(spart)) # part
# print(type(snote_array)) # ndarray

# print(dir(spart))
# print([i for i in dir(spart) if 'measure' in i])

# print(type(spart.notes)) # list
# print(dir(spart.notes[0])) # list

# print(type(spart.measures)) # list
# print(type(spart.measures[0])) # Measure
# print(dir(spart.measures[0])) # 

for measure in spart.measures[:10]:
    # print(type(measure.number))
    print(measure.number)

for note in spart.notes:
    print(f"Note {note.id} starts in measure {spart.measure_number_map(note.start.t)}") # breaks
    # print(f"Note {note.id} starts is contained in beats {spart.measure_map(note.start.t)}") # works
    


# print(spart.measure_map)
# print(spart.measure_number_map)
# print(spart.measures)


# def create_measures_notes_dict(score_path, piece):
    
#     measures_notes_dict = {}
#     musicxml_file = os.path.join(score_path, 'K%s-%s.musicxml' % (piece[:3], piece[-1]))
#     root = ET.parse(musicxml_file)
    
#     for measure in root.findall("//measure"):
#         measure_number = measure.get('number')
#         notes = measure.findall('note')
#         notes_id = [note.get('id') for note in notes]
#         # measures can be organised within parts, or parts within measures.
#         # handles the case where the same measures are split between different parts
#         if measure_number in measures_notes_dict:
#             measures_notes_dict[measure_number].extend(notes_id)
#         else:
#             measures_notes_dict[measure_number] = notes_id

#     return measures_notes_dict
    
