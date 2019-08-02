# encoding: utf-8
# pylint: skip-file
"""
This module contains tests.
"""

import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')
MUSICXML_PATH = os.path.join(DATA_PATH, 'musicxml')

# this is a list of files for which importing and subsequent exporting should
# yield identical MusicXML
MUSICXML_IMPORT_EXPORT_TESTFILES = [os.path.join(MUSICXML_PATH, fn) for fn in
                                    ['test_note_ties.xml',
                                     'test_note_ties_divs.xml']]
