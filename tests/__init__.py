# encoding: utf-8
# pylint: skip-file
"""
This module contains tests.
"""

import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
MUSICXML_PATH = os.path.join(DATA_PATH, "musicxml")
MEI_PATH = os.path.join(DATA_PATH, "mei")
MATCH_PATH = os.path.join(DATA_PATH, "match")
NAKAMURA_PATH = os.path.join(DATA_PATH, "nakamura")

# this is a list of files for which importing and subsequent exporting should
# yield identical MusicXML
MUSICXML_IMPORT_EXPORT_TESTFILES = [
    os.path.join(MUSICXML_PATH, fn)
    for fn in ["test_note_ties.xml", "test_note_ties_divs.xml"]
]
MUSICXML_UNFOLD_TESTPAIRS = [
    (
        os.path.join(MUSICXML_PATH, fn1),
        os.path.join(MUSICXML_PATH, fn2),
        os.path.join(MUSICXML_PATH, fn3),
    )
    for fn1, fn2, fn3 in [
        (
            "test_unfold_timeline.xml",
            "test_unfold_timeline_result.xml",
            "test_unfold_timeline_result_updated_ids.xml",
        )
    ]
]

# This is a list of files for testing Chew and Wu's VOSA. (More files to come?)
VOSA_TESTFILES = [
    os.path.join(MUSICXML_PATH, fn) for fn in ["test_chew_vosa_example.xml"]
]


MATCH_IMPORT_EXPORT_TESTFILES = [
    os.path.join(MATCH_PATH, fn) for fn in ["test_fuer_elise.match"]
]

# This is a list of files for testing Nakamura et al.'s corresp and match file loading
NAKAMURA_IMPORT_TESTFILES = [
    os.path.join(NAKAMURA_PATH, fn)
    for fn in [
        "Shi05_infer_corresp.txt",
        "Shi05_infer_corresp.txt",
        "test_nakamura_performance_corresp.txt",
        "test_nakamura_performance_match.txt",
    ]
]

METRICAL_POSITION_TESTFILES = [
    os.path.join(MUSICXML_PATH, fn)
    for fn in ["test_metrical_position.xml", "test_anacrusis.xml"]
]

NOTE_ARRAY_TESTFILES = [os.path.join(MUSICXML_PATH, fn) for fn in ["test_beats.xml"]]

MEI_TESTFILES = [
    os.path.join(MEI_PATH, fn)
    for fn in [
        "example_noMeasures_noBeams.mei",
        "example_noMeasures_withBeams.mei",
        "example_withMeasures_noBeams.mei",
        "example_withMeasures_withBeams.mei",
        "Bach_Prelude.mei",
        "Schubert_An_die_Sonne_D.439.mei",
        "test_tuplets.mei",
        "test_ties.mei",
        "test_metrical_position.mei",
        "test_clefs_tss.mei",
        "test_grace_note.mei",
    ]
]
