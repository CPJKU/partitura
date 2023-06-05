#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: skip-file
"""
This module contains tests.
"""

import os
import glob

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
MUSICXML_PATH = os.path.join(DATA_PATH, "musicxml")
MEI_PATH = os.path.join(DATA_PATH, "mei")
KERN_PATH = os.path.join(DATA_PATH, "kern")
MATCH_PATH = os.path.join(DATA_PATH, "match")
NAKAMURA_PATH = os.path.join(DATA_PATH, "nakamura")
MIDI_PATH = os.path.join(DATA_PATH, "midi")
PARANGONADA_PATH = os.path.join(DATA_PATH, "parangonada")
WAV_PATH = os.path.join(DATA_PATH, "wav")
PNG_PATH = os.path.join(DATA_PATH, "png")

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

MUSICXML_UNFOLD_COMPLEX = [
    (
        os.path.join(MUSICXML_PATH, fn1),
        os.path.join(MUSICXML_PATH, fn2),
    )
    for fn1, fn2 in [("test_unfold_complex.xml", "test_unfold_complex_result.xml")]
]


MUSICXML_NOTE_FEATURES = [
    os.path.join(MUSICXML_PATH, fn) for fn in ["test_note_features.xml"]
]

MUSICXML_UNFOLD_VOLTA = [
    (
        os.path.join(MUSICXML_PATH, fn1),
        os.path.join(MUSICXML_PATH, fn2),
    )
    for fn1, fn2 in [
        ("test_unfold_volta_numbers.xml", "test_unfold_volta_numbers_result.xml")
    ]
]

MUSICXML_UNFOLD_DACAPO = [
    (
        os.path.join(MUSICXML_PATH, fn1),
        os.path.join(MUSICXML_PATH, fn2),
    )
    for fn1, fn2 in [("test_unfold_dacapo.xml", "test_unfold_dacapo_result.xml")]
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

TIME_SIGNATURE_MAP_EDGECASES_TESTFILES = [
    os.path.join(MUSICXML_PATH, fn)
    for fn in ["test_ts_map_ts_starts_not_at_zero.xml"]
]

REST_ARRAY_TESTFILES = [
    os.path.join(MUSICXML_PATH, fn)
    for fn in ["test_unfold_complex.xml", "test_rest.musicxml"]
]

NOTE_ARRAY_TESTFILES = [os.path.join(MUSICXML_PATH, fn) for fn in ["test_beats.xml"]]

OCTAVE_SHIFT_TESTFILES = [os.path.join(MUSICXML_PATH, fn) for fn in ["example_octave_shift.musicxml"]]

MERGE_PARTS_TESTFILES = [
    os.path.join(MUSICXML_PATH, fn)
    for fn in [
        "test_part_group.xml",
        "test_multi_part.xml",
        "test_multi_part_change_divs.xml",
        "test_metrical_position.xml",
        "test_merge_interpolation.xml",
        "test_single_part_change_divs.xml",
        "test_merge_voices1.xml",
        "test_merge_voices2.xml",
    ]
]

PIANOROLL_TESTFILES = [
    os.path.join(MUSICXML_PATH, fn)
    for fn in [
        "test_length_pianoroll.xml",
        "test_pianoroll_sum.xml",
        "test_pianoroll_sum_reduced.xml",
    ]
]


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
        "Beethoven_Op119_Nr01-Breitkopf.mei",
        "test_tuplets_no_ppq.mei",
        "Beethoven_Op119_Nr02-Breitkopf.mei",
        "test_parts_duration.mei",
        "test_parts_duration2.mei",
        "test_barline.mei",
        "test_unfold_complex.mei",
        "test_articulation.mei",
        "test_merge_voices2.mei",
        "CRIM_Mass_0030_4.mei",
    ]
]

KERN_TESTFILES = [
    os.path.join(KERN_PATH, fn)
    for fn in [
        "single_voice_example.krn",
        "long_example.krn",
        "double_repeat_example.krn",
        "fine_with_repeat.krn",
        "tuple_durations.krn",
        "voice_dublifications.krn",
        "variable_length_pr_bug.krn",
        "chor228.krn",
    ]
]

KERN_TIES = [os.path.join(KERN_PATH, fn) for fn in ["tie_mismatch.krn"]]
M21_TESTFILES = [
    os.path.join(DATA_PATH, "musicxml", fn)
    for fn in [
        "test_clefs_tss.xml",
        "test_grace_note.xml",
        "test_chew_vosa_example.xml",
        "test_note_ties.xml",
    ]
]
HARMONY_TESTFILES = [os.path.join(MUSICXML_PATH, fn) for fn in ["test_harmony.musicxml"]]

MOZART_VARIATION_FILES = dict(
    musicxml=os.path.join(MUSICXML_PATH, "mozart_k265_var1.musicxml"),
    midi=os.path.join(MIDI_PATH, "mozart_k265_var1.mid"),
    match=os.path.join(MATCH_PATH, "mozart_k265_var1.match"),
    parangonada_align=os.path.join(PARANGONADA_PATH, "mozart_k265_var1", "align.csv"),
    parangonada_feature=os.path.join(
        PARANGONADA_PATH, "mozart_k265_var1", "feature.csv"
    ),
    parangonada_spart=os.path.join(PARANGONADA_PATH, "mozart_k265_var1", "part.csv"),
    parangonada_ppart=os.path.join(PARANGONADA_PATH, "mozart_k265_var1", "ppart.csv"),
    parangonada_zalign=os.path.join(PARANGONADA_PATH, "mozart_k265_var1", "zalign.csv"),
)


WAV_TESTFILES = [
    os.path.join(WAV_PATH, fn)
    for fn in [
        "example_linear_equal_temperament_sr8000.wav",
    ]
]

PNG_TESTFILES = glob.glob(os.path.join(PNG_PATH, "*.png"))
