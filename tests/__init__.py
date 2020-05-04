# encoding: utf-8
# pylint: skip-file
"""
This module contains tests.
"""

import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')
MUSICXML_PATH = os.path.join(DATA_PATH, 'musicxml')
MATCH_PATH = os.path.join(DATA_PATH, 'match')
NAKAMURA_PATH = os.path.join(DATA_PATH, 'nakamuramatch')

# this is a list of files for which importing and subsequent exporting should
# yield identical MusicXML
MUSICXML_IMPORT_EXPORT_TESTFILES = [os.path.join(MUSICXML_PATH, fn) for fn in
                                    ['test_note_ties.xml',
                                     'test_note_ties_divs.xml']]
MUSICXML_UNFOLD_TESTPAIRS = [(os.path.join(MUSICXML_PATH, fn1), os.path.join(MUSICXML_PATH, fn2))
                             for fn1, fn2 in
                             [('test_unfold_timeline.xml', 'test_unfold_timeline_result.xml')]]

# This is a list of files for testing Chew and Wu's VOSA. (More files to come?)
VOSA_TESTFILES = [os.path.join(MUSICXML_PATH, fn) for fn in
                  ['test_chew_vosa_example.xml']]


MATCH_IMPORT_EXPORT_TESTFILES = [os.path.join(MATCH_PATH, fn) for fn in ['test_fuer_elise.match']]

<<<<<<< HEAD
NAKAMURA_PATH_IMPORT_TESTFILES = [(os.path.join(NAKAMURA_PATH, fn) for fn in piece)
                                  for piece in [('chopin_op23_corresp.txt', 'chopin_op23_performance.mid',
                                                 'chopin_op23_score.mid', 'chopin_op23.musicxml')]]
=======
# NAKAMURA_PATH_IMPORT_TESTFILES = [(os.path.join(NAKAMURA_PATH, fn) for fn in piece)
#                                   for piece in [('test_corresp.txt', 'test_performance.mid',
#                                                  'test_score.mid', 'chopin_op23.musicxml')]]
>>>>>>> develop
