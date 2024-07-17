#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains the test cases for testing the note_array attribute of
the Part class.
"""

import unittest

import partitura.score as score
from partitura import load_musicxml, load_kern, load_score
from partitura.utils.music import note_array_from_part, ensure_notearray
from partitura.musicanalysis import note_array_to_score
import numpy as np

from tests import NOTE_ARRAY_TESTFILES, KERN_TESTFILES, METRICAL_POSITION_TESTFILES


class TestNoteArray(unittest.TestCase):
    """
    Test the note_array attribute of the Part class
    """

    def test_notearray_1(self):
        part = score.Part("P0", "My Part")

        part.set_quarter_duration(0, 10)
        part.add(score.TimeSignature(3, 4), start=0)
        part.add(score.Note(id="n0", step="A", octave=4), start=0, end=10)

        note_array = part.note_array()
        self.assertTrue(len(note_array) == 1)

    def test_notearray_beats(self):
        score = load_musicxml(NOTE_ARRAY_TESTFILES[0])[0]
        note_array = score.note_array()
        expected_onset_beats = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29, 30, 32]

        self.assertTrue(np.array_equal(note_array["onset_beat"], expected_onset_beats))

    def test_notearray_musical_beats1(self):
        score = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        score[0].use_musical_beat()
        note_array = note_array_from_part(score[0])
        expected_onset_beats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14]

        self.assertTrue(np.array_equal(note_array["onset_beat"], expected_onset_beats))
        self.assertTrue(
            np.array_equal(score[0].note_array()["onset_beat"], expected_onset_beats)
        )

    def test_use_musical_beats1(self):
        score = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        score[0].use_musical_beat({"6/8": 3, "2/4": 1})
        note_array = note_array_from_part(score[0])
        expected_onset_beats = [0, 1.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11.5, 12.5]

        self.assertTrue(np.array_equal(note_array["onset_beat"], expected_onset_beats))
        self.assertTrue(
            np.array_equal(score[0].note_array(), note_array_from_part(score[0]))
        )

    def test_use_musical_beats2(self):
        score = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        # set musical beats
        score[0].use_musical_beat()
        note_array = score[0].note_array()
        self.assertTrue(score[0]._use_musical_beat == True)
        # unset musical beats
        score[0].use_notated_beat()
        self.assertTrue(score[0]._use_musical_beat == False)
        self.assertFalse(np.array_equal(score[0].note_array(), note_array))

    def test_note_array_to_score_multiple_ts(self):
        score = load_musicxml(NOTE_ARRAY_TESTFILES[0])
        note_array = score.note_array(include_time_signature=True)
        new_score = note_array_to_score(note_array)
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

    def test_note_array_to_score_anacrusis(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[1])
        note_array = score.note_array(include_time_signature=True)
        new_score = note_array_to_score(note_array)
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

    def test_note_array_to_score_only_divs(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[1])
        note_array = score.note_array(include_time_signature=True)
        for div in [1,2,4,8]:
            new_score = note_array_to_score(note_array[["onset_div", "duration_div", "pitch"]],divs = div)
            new_note_array = new_score.note_array(include_time_signature=True)
            self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_div"]/div))
            self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_div"]/div))

    def test_note_array_to_score_only_div_time_sig(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[1])
        note_array = score.note_array(include_time_signature=True)
        for time_sig in [4,8,9]:
            new_score = note_array_to_score(note_array[["onset_div", "duration_div", "pitch"]],divs = 1, time_sigs = [(2,4, time_sig)])
            new_note_array = new_score.note_array(include_time_signature=True)
            self.assertTrue(np.all(new_note_array["ts_beat_type"] == np.ones_like(note_array["onset_div"])*time_sig))

    def test_note_array_to_score_only_beats(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[1])
        note_array = score.note_array(include_time_signature=True)
        # without time signature -> barebones score without measures/anacrusis
        new_score = note_array_to_score(note_array[["onset_beat", "duration_beat", "pitch"]])
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]+1.0))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

        # with time signature -> measures/anacrusis kept, but div/beat ratio might change
        new_score = note_array_to_score(note_array[["onset_beat", "duration_beat", "pitch"]], time_sigs= [(1,3,8)] )
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]*2))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]*2))

        # with default time signature -> measures/anacrusis kept, 4/4 corrsponds to original
        new_score = note_array_to_score(note_array[["onset_beat", "duration_beat", "pitch"]], estimate_time = True)
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

    def test_note_array_to_score_beats_and_div(self):
        score = load_musicxml(METRICAL_POSITION_TESTFILES[1])
        note_array = score.note_array(include_time_signature=True)
        # without time signature -> barebones score without measures/anacrusis
        new_score = note_array_to_score(note_array[["onset_div","duration_div","onset_beat", "duration_beat", "pitch"]])
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]+1.0))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

        # with time signature -> measures/anacrusis kept, but div/beat ratio might change
        new_score = note_array_to_score(note_array[["onset_div","duration_div","onset_beat", "duration_beat", "pitch"]], time_sigs= [(1,3,8)] )
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]*2))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]*2))

        # with default time signature -> measures/anacrusis kept, 4/4 corrsponds to original
        new_score = note_array_to_score(note_array[["onset_div","duration_div","onset_beat", "duration_beat", "pitch"]], estimate_time = True)
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))

        # with original time signature -> measures/anacrusis kept, other arguments overridden
        new_score = note_array_to_score(note_array[["onset_div","duration_div","onset_beat", "duration_beat", "pitch", "ts_beats", "ts_beat_type"]] , time_sigs= [(1,3,8)], estimate_time = True)
        new_note_array = new_score.note_array(include_time_signature=True)
        self.assertTrue(np.all(new_note_array["onset_beat"] == note_array["onset_beat"]))
        self.assertTrue(np.all(new_note_array["duration_beat"] == note_array["duration_beat"]))


    def test_notearray_ts_beats(self):
        part = load_musicxml(NOTE_ARRAY_TESTFILES[0])[0]
        note_array = note_array_from_part(part, include_time_signature=True)
        expected_beats = [6, 6, 9, 9, 9, 12, 12, 12, 12, 2, 2, 2, 2]
        self.assertTrue(np.array_equal(note_array["ts_beats"], expected_beats))

        # now using musical beats
        part.use_musical_beat()
        note_array = note_array_from_part(part, include_time_signature=True)
        expected_musical_beats = [2, 2, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2]
        self.assertTrue(
            np.array_equal(note_array["ts_mus_beats"], expected_musical_beats)
        )

    def test_ensure_na_different_divs(self):
        # check if divs are correctly rescaled when producing a note array from
        # parts with different divs values
        parts = load_kern(KERN_TESTFILES[7]).parts
        merged_note_array = ensure_notearray(parts)
        for note in merged_note_array[-4:]:
            self.assertTrue(note["onset_div"] == 2208)
            self.assertTrue(note["duration_div"] == 96)
            self.assertTrue(note["divs_pq"] == 96)

    def test_score_notearray_method(self):
        """
        Test that note array generated from the Score class method
        include all relevant information.
        """

        for fn in NOTE_ARRAY_TESTFILES:

            scr = load_score(fn)

            na = scr.note_array(
                include_pitch_spelling=True,
                include_key_signature=True,
                include_time_signature=True,
                include_grace_notes=True,
                include_metrical_position=True,
                include_staff=True,
                include_divs_per_quarter=True,
            )

            expected_field_names = [
                "onset_beat",
                "duration_beat",
                "onset_quarter",
                "duration_quarter",
                "onset_div",
                "duration_div",
                "pitch",
                "voice",
                "id",
                "step",
                "alter",
                "octave",
                "is_grace",
                "grace_type",
                "ks_fifths",
                "ks_mode",
                "ts_beats",
                "ts_beat_type",
                "ts_mus_beats",
                "is_downbeat",
                "rel_onset_div",
                "tot_measure_div",
                "staff",
                "divs_pq",
            ]

            for field_name in expected_field_names:
                # check that the note array contain the relevant
                # field.
                self.assertTrue(field_name in na.dtype.names)


if __name__ == "__main__":
    unittest.main()
