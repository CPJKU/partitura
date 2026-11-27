#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for Matchfile import
"""
import unittest
import numpy as np
import re
import os
from tempfile import TemporaryDirectory

from tests import MOZART_VARIATION_FILES

from partitura.io.exportmatch import matchfile_from_alignment, save_match
from partitura.io.importmatch import load_match
from partitura.io.matchfile_utils import Version
from partitura.io.matchlines_v1 import MatchSection, MatchOmittedSection
from partitura import load_score
from partitura.performance import PerformedPart


class TestExportMatch(unittest.TestCase):
    def test_matchfile_from_alignment(self):
        """
        test `matchfile_from_alignment`
        """
        score_fn = MOZART_VARIATION_FILES["musicxml"]

        score = load_score(score_fn)
        spart = score[0]
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, alignment = load_match(match_fn)

        matchfile = matchfile_from_alignment(
            alignment=alignment,
            ppart=performance[0],
            spart=spart,
            assume_part_unfolded=True,
        )

        sna = spart.note_array()
        pna = performance.note_array()

        # assert that matchfile contains the same number of notes
        self.assertTrue(len(sna) == len(matchfile.snotes))
        self.assertTrue(len(pna) == len(matchfile.notes))

        snote_ids = [n.Anchor for n in matchfile.snotes]
        pnote_ids = [n.Id for n in matchfile.notes]
        # assert that all snotes in the matchfile are in the note array
        self.assertTrue(all([n.Anchor in sna["id"] for n in matchfile.snotes]))

        # assert that all notes in the score are in the matchfile
        self.assertTrue(all([nid in snote_ids for nid in sna["id"]]))

        # assert that all notes in the matchfile are in the note array
        self.assertTrue(all([n.Id in pna["id"] for n in matchfile.notes]))

        # assert that all notes in the performance are in the matchfile
        self.assertTrue(all(nid in pnote_ids for nid in pna["id"]))

        for ml in matchfile.lines:
            self.assertTrue(isinstance(ml.matchline, str))

    def test_save_match(self):
        """
        Test save_match
        """
        score_fn = MOZART_VARIATION_FILES["musicxml"]

        score = load_score(score_fn)
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, alignment = load_match(match_fn)
        pna1 = performance.note_array()
        with TemporaryDirectory() as tmpdir:

            out = os.path.join(tmpdir, "test.match")
            save_match(
                alignment=alignment,
                performance_data=performance,
                score_data=score,
                out=out,
                performer="A Pianist",
                composer="W. A. Mozart",
                piece="mozart_k265_var1",
                score_filename=os.path.basename(score_fn),
                performance_filename=os.path.basename(MOZART_VARIATION_FILES["midi"]),
                assume_unfolded=True,
            )

            perf_from_saved_match, alignment_from_saved_match = load_match(out)

            # Test Metadata
            # Assuming perf_from_saved_match or the parsed mf holds info
            # You may need to load the MatchFile object directly to test info lines easily
            from partitura.io.importmatch import load_matchfile
            mf = load_matchfile(out)
            self.assertEqual(mf.info("composer"), "W. A. Mozart")
            self.assertEqual(mf.info("performer"), "A Pianist")
            self.assertEqual(mf.info("piece"), "mozart_k265_var1")

            # Test Pedal Data
            pedals_original = performance[0].controls
            pedals_saved = perf_from_saved_match[0].controls
            self.assertEqual(len(pedals_original), len(pedals_saved))
            for p1, p2 in zip(pedals_original, pedals_saved):
                self.assertEqual(p1["time"], p2["time"])
                self.assertEqual(p1["value"], p2["value"])

        pna2 = perf_from_saved_match.note_array()

        # Test that performance data is the same
        for field in (
            "onset_sec",
            "duration_sec",
            "onset_tick",
            "duration_tick",
            "pitch",
            "velocity",
        ):
            self.assertTrue(np.allclose(pna2[field], pna1[field]))

        # Test that the alignment info is correct
        for al in alignment_from_saved_match:
            self.assertTrue(al in alignment)

        for al in alignment:
            self.assertTrue(al in alignment_from_saved_match)

    def test_save_match_old_and_new_versions(self):
        """
        Test save_match for version 1.0.0 and a newer version with sections.
        """
        score_fn = MOZART_VARIATION_FILES["musicxml"]
        score = load_score(score_fn)
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, alignment = load_match(match_fn)

        section = {
            "id": "section_1",
            "start_in_beats_unfolded": 0.0,
            "end_in_beats_unfolded": 1.0,
            "start_in_beats_original": 0.0,
            "end_in_beats_original": 1.0,
            "start_in_perf_time": 0.0,
            "end_in_perf_time": 1.0,
            "section_attr_list": ["section_attr"],
        }
        omitted_section = {
            "id": "omitted_1",
            "start_in_beats_unfolded": 1.0,
            "end_in_beats_unfolded": 2.0,
            "start_in_beats_original": 1.0,
            "end_in_beats_original": 2.0,
            "section_attr_list": ["omitted_attr"],
        }

        with TemporaryDirectory() as tmpdir:
            old_out = os.path.join(tmpdir, "old_version.match")
            save_match(
                alignment=alignment,
                performance_data=performance,
                score_data=score,
                out=old_out,
                version=(1, 0, 0),
                sections=[section],
                omitted_sections=[omitted_section],
            )
            with open(old_out, "r", encoding="utf-8") as fh:
                old_content = fh.read()

            self.assertIn("matchFileVersion", old_content)
            self.assertNotIn("section_1", old_content)
            self.assertNotIn("omitted_1", old_content)

            new_out = os.path.join(tmpdir, "new_version.match")
            save_match(
                alignment=alignment,
                performance_data=performance,
                score_data=score,
                out=new_out,
                version=(1, 1, 0),
                sections=[section],
                omitted_sections=[omitted_section],
            )
            with open(new_out, "r", encoding="utf-8") as fh:
                new_content = fh.read()

            self.assertIn("section_1", new_content)
            self.assertIn("omitted_1", new_content)

    def test_matchfile_from_alignment_version_specific_lines(self):
        """
        Test that version 1.0.0 ignores section lines, while newer versions emit them.
        """
        score_fn = MOZART_VARIATION_FILES["musicxml"]
        score = load_score(score_fn)
        spart = score[0]
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, alignment = load_match(match_fn)

        section = {
            "id": "section_1",
            "start_in_beats_unfolded": 0.0,
            "end_in_beats_unfolded": 1.0,
            "start_in_beats_original": 0.0,
            "end_in_beats_original": 1.0,
            "start_in_perf_time": 0.0,
            "end_in_perf_time": 1.0,
            "section_attr_list": ["section_attr"],
        }
        omitted_section = {
            "id": "omitted_1",
            "start_in_beats_unfolded": 1.0,
            "end_in_beats_unfolded": 2.0,
            "start_in_beats_original": 1.0,
            "end_in_beats_original": 2.0,
            "section_attr_list": ["omitted_attr"],
        }

        mf_old = matchfile_from_alignment(
            alignment=alignment,
            ppart=performance[0],
            spart=spart,
            version=Version(1, 0, 0),
            sections=[section],
            omitted_sections=[omitted_section],
        )
        self.assertFalse(
            any(
                isinstance(line, (MatchSection, MatchOmittedSection))
                for line in mf_old.lines
            )
        )

        mf_new = matchfile_from_alignment(
            alignment=alignment,
            ppart=performance[0],
            spart=spart,
            version=Version(1, 1, 0),
            sections=[section],
            omitted_sections=[omitted_section],
        )
        self.assertTrue(any(isinstance(line, MatchSection) for line in mf_new.lines))
        self.assertTrue(
            any(isinstance(line, MatchOmittedSection) for line in mf_new.lines)
        )
    
    def test_virtual_notes_export_version_1_1_0(self):
        """
        Test the export of virtual notes introduced in version 1.1.0.
        Virtual notes occur implicitly when the same score_id or performance_id 
        is aligned multiple times.
        """
        from partitura.io.matchlines_v1 import (
            MatchSnoteNote,
            MatchVirtualSnoteNote,
            MatchSnoteVirtualNote,
            MatchVirtualSnoteVirtualNote
        )
        from partitura.io.importmatch import load_matchfile

        score_fn = MOZART_VARIATION_FILES["musicxml"]
        score = load_score(score_fn)
        spart = score[0]
        match_fn = MOZART_VARIATION_FILES["match"]
        performance, original_alignment = load_match(match_fn)
        
        # Extract two distinct valid matches to get authentic IDs
        matches = [a for a in original_alignment if a["label"] == "match"]
        m1, m2 = matches[0], matches[1]
        
        s1, p1 = m1["score_id"], m1["performance_id"]
        s2, p2 = m2["score_id"], m2["performance_id"]
        
        # Create a synthetic alignment to force the exporter into all 4 states:
        synthetic_alignment = [
            # 1. Standard (neither seen before) -> MatchSnoteNote
            {"label": "match", "score_id": s1, "performance_id": p1}, 
            
            # 2. Score ID seen, Perf ID new -> MatchVirtualSnoteNote
            {"label": "match", "score_id": s1, "performance_id": p2}, 
            
            # 3. Score ID new, Perf ID seen -> MatchSnoteVirtualNote
            {"label": "match", "score_id": s2, "performance_id": p1}, 
            
            # 4. Both IDs seen before -> MatchVirtualSnoteVirtualNote
            {"label": "match", "score_id": s1, "performance_id": p1}, 
        ]
        
        # Export using 1.1.0 logic
        mf = matchfile_from_alignment(
            alignment=synthetic_alignment,
            ppart=performance[0],
            spart=spart,
            version=Version(1, 1, 0),
            assume_part_unfolded=True
        )
        
        # Extract the classes of the generated lines to verify the routing logic worked
        line_types = [type(line) for line in mf.lines]
        
        self.assertIn(MatchSnoteNote, line_types, "Standard MatchSnoteNote missing")
        self.assertIn(MatchVirtualSnoteNote, line_types, "MatchVirtualSnoteNote missing")
        self.assertIn(MatchSnoteVirtualNote, line_types, "MatchSnoteVirtualNote missing")
        self.assertIn(MatchVirtualSnoteVirtualNote, line_types, "MatchVirtualSnoteVirtualNote missing")
        
        # Perform a round-trip IO check to ensure the parsers successfully load these lines back
        with TemporaryDirectory() as tmpdir:
            out_fn = os.path.join(tmpdir, "virtual_test.match")
            mf.write(out_fn)
            
            parsed_mf = load_matchfile(out_fn)
            parsed_types = [type(line) for line in parsed_mf.lines]
            
            self.assertIn(MatchVirtualSnoteNote, parsed_types, "Failed to parse MatchVirtualSnoteNote")
            self.assertIn(MatchSnoteVirtualNote, parsed_types, "Failed to parse MatchSnoteVirtualNote")
            self.assertIn(MatchVirtualSnoteVirtualNote, parsed_types, "Failed to parse MatchVirtualSnoteVirtualNote")

    def test_non_unique_alignment_round_trip(self):
        """
        Test that nothing is lost when loading and saving match files
        with virtual notes and sections.
        """
        score = load_score(MOZART_VARIATION_FILES["musicxml"])
        performance, alignment = load_match(MOZART_VARIATION_FILES["match"])
        matches = [a for a in alignment if a["label"] == "match"]
        sids = [a["score_id"] for a in matches]
        pids = [a["performance_id"] for a in matches]

        new_alignment = [
            dict(a, score_attributes_list=["section_s1"]) for a in matches[4:]
        ]
        new_alignment += [a for a in alignment if a["label"] != "match"]
        new_alignment += [
            # score notes with several performance notes. The first performance
            # notes are virtual, so these lines are written before the
            # line specifying the score note if sorting by time.
            dict(label="match", score_id=sids[0], performance_id=pids[3]),
            dict(
                label="match",
                score_id=sids[0],
                performance_id=pids[0],
                score_attributes_list=["rehearsal_repetition", "pitch_error"],
            ),
            dict(label="match", score_id=sids[0], performance_id=pids[1]),
            # performance notes with several score notes
            dict(
                label="match",
                score_id=sids[1],
                performance_id=pids[1],
                score_attributes_list=["musically_identical_to_score_id_x"],
            ),
            # both
            dict(
                label="match",
                score_id=sids[1],
                performance_id=pids[0],
                score_attributes_list=["sustained_realization"],
            ),
            dict(
                label="deletion",
                score_id=sids[2],
                score_attributes_list=["not_played"],
            ),
            dict(label="deletion", score_id=sids[3]),
            dict(label="insertion", performance_id=pids[2]),
        ]
        sections = [
            dict(
                id="s1",
                start_in_beats_unfolded=0.0,
                end_in_beats_unfolded=4.0,
                start_in_beats_original=0.0,
                end_in_beats_original=4.0,
                start_in_perf_time=10,
                end_in_perf_time=2000,
                section_attr_list=["rehearsal", "left_hand"],
            ),
            dict(
                id="s2",
                start_in_beats_unfolded=2.0,
                end_in_beats_unfolded=4.0,
                start_in_beats_original=2.0,
                end_in_beats_original=4.0,
                start_in_perf_time=2000,
                end_in_perf_time=3000,
                section_attr_list=[],
            ),
        ]
        omitted_sections = [
            dict(
                id="os1",
                start_in_beats_unfolded=100.0,
                end_in_beats_unfolded=120.0,
                start_in_beats_original=50.0,
                end_in_beats_original=70.0,
                section_attr_list=["not_rehearsed"],
            )
        ]

        def pairs(alignment, *info):
            return sorted(
                [
                    (al["label"], al.get("score_id", ""), al.get("performance_id", ""))
                    + tuple(x)
                    for al, *x in zip(alignment, *info)
                ]
            )

        def read(fn):
            with open(fn) as f:
                return f.read()

        for sort_alignment in ("ptime", "stime", False):
            with TemporaryDirectory() as tmpdir:
                out = os.path.join(tmpdir, "first.match")
                save_match(
                    new_alignment,
                    performance,
                    score,
                    out=out,
                    sections=sections,
                    omitted_sections=omitted_sections,
                    sort_alignment=sort_alignment,
                )
                loaded_perf, loaded_alignment, info = load_match(
                    out, return_alignment_info=True
                )
                note_types, score_attributes = info.note_types, info.score_attributes
                loaded_sections, loaded_omitted = info.sections, info.omitted_sections

                self.assertEqual(
                    len(loaded_perf.note_array()), len(performance.note_array())
                )
                self.assertEqual(pairs(loaded_alignment), pairs(new_alignment))
                self.assertEqual(loaded_sections, sections)
                self.assertEqual(loaded_omitted, omitted_sections)

                info_by_ids = {
                    (al.get("score_id"), al.get("performance_id")): (nt, sa)
                    for al, nt, sa in zip(
                        loaded_alignment, note_types, score_attributes
                    )
                }
                for al, note_type in zip(
                    new_alignment[-8:],
                    [
                        ("snote", "note"),
                        ("virtualSnote", "note"),
                        ("virtualSnote", "note"),
                        ("snote", "virtualPnote"),
                        ("virtualSnote", "virtualPnote"),
                        ("snote", "-"),
                        ("snote", "-"),
                        ("-", "note"),
                    ],
                ):
                    nt, sa = info_by_ids[(al.get("score_id"), al.get("performance_id"))]
                    self.assertEqual(nt, note_type)
                    attributes = al.get("score_attributes_list", [])
                    if note_type[0] == "virtualSnote":
                        self.assertEqual(sa, attributes)
                    elif note_type[0] == "snote":
                        self.assertTrue(set(attributes) < set(sa))

                # and twice more, with the attributes in the alignment. The
                # first match of a note in the alignment gets the full line,
                # which changes here when sorting by time (see above).
                previous, saved = out, []
                for i in range(2):
                    loaded_perf, loaded_alignment, info = load_match(
                        previous, return_alignment_info=True
                    )
                    for al, attributes in zip(loaded_alignment, info.score_attributes):
                        al["score_attributes_list"] = attributes
                    previous = os.path.join(tmpdir, f"again_{i}.match")
                    save_match(
                        loaded_alignment,
                        loaded_perf,
                        score,
                        out=previous,
                        sort_alignment=sort_alignment,
                        sections=info.sections,
                        omitted_sections=info.omitted_sections,
                    )
                    self.assertEqual(
                        pairs(load_match(previous)[1]), pairs(new_alignment)
                    )
                    saved.append(read(previous))
                self.assertEqual(saved[0], saved[1])
                if not sort_alignment:
                    self.assertEqual(read(out), saved[0])

    def test_virtual_notes(self):
        """
        Test that later matches of the same notes are virtual
        """
        score = load_score(MOZART_VARIATION_FILES["musicxml"])
        performance, alignment = load_match(MOZART_VARIATION_FILES["match"])
        matches = [a for a in alignment if a["label"] == "match"]
        (s1, s2), (p1, p2) = [
            [a[key] for a in matches[:2]] for key in ("score_id", "performance_id")
        ]
        new_alignment = [
            dict(label="match", score_id=sid, performance_id=pid)
            for sid, pid in ((s1, p1), (s2, p1), (s1, p2), (s2, p2))
        ]

        def get_types(alignment, **kwargs):
            mf = matchfile_from_alignment(
                alignment=alignment,
                ppart=performance[0],
                spart=score[0],
                sort_alignment=False,
                **kwargs,
            )
            names = dict(
                MatchSnote="snote",
                MatchVirtualSnote="virtualSnote",
                MatchNote="note",
                MatchVirtualPNote="virtualPnote",
            )
            return [
                (names[type(ln.snote).__name__], names[type(ln.note).__name__])
                for ln in mf.lines
                if hasattr(ln, "snote")
            ]

        first_come = [
            ("snote", "note"),
            ("snote", "virtualPnote"),
            ("virtualSnote", "note"),
            ("virtualSnote", "virtualPnote"),
        ]
        # the order in the alignment decides which match is the first one
        self.assertEqual(get_types(new_alignment), first_come)
        self.assertEqual(get_types(new_alignment[::-1]), first_come)

        # older versions do not have virtual notes
        with self.assertRaises(ValueError):
            get_types(new_alignment, version=Version(1, 0, 0))
        with self.assertRaises(ValueError):
            save_match(new_alignment, performance, score, version=(1, 0, 0))
        # but unique alignments are fine
        self.assertEqual(
            len(save_match(new_alignment[:1], performance, score, version=(1, 0, 0)).lines),
            len(save_match(new_alignment[:1], performance, score).lines),
        )


    def test_deletions_in_omitted_sections(self):
        """
        Test that deletions are only left out if they are in an omitted section
        (including its start, but not its end).
        """
        score = load_score(MOZART_VARIATION_FILES["musicxml"])
        performance, alignment = load_match(MOZART_VARIATION_FILES["match"])
        sna = score.note_array()

        start, end = np.unique(sna["onset_beat"])[[4, 12]]
        omitted = sna["id"][(sna["onset_beat"] >= start) & (sna["onset_beat"] < end)]
        at_end = sna["id"][sna["onset_beat"] == end]
        self.assertTrue(np.any(sna["onset_beat"] == start))

        # nothing but the first 4 notes was performed
        matched = [a for a in alignment if a["label"] == "match"][:4]
        matched_ids = [a["score_id"] for a in matched]
        self.assertFalse(np.any(np.isin(matched_ids, omitted)))
        new_alignment = matched + [
            dict(label="deletion", score_id=sid)
            for sid in sna["id"]
            if sid not in matched_ids
        ]

        def get_deletions(omitted_sections):
            mf = matchfile_from_alignment(
                alignment=new_alignment,
                ppart=performance[0],
                spart=score[0],
                omitted_sections=omitted_sections,
            )
            return set(n.Anchor for n in mf.deletions)

        all_deletions = get_deletions([])
        self.assertEqual(len(all_deletions), len(sna) - 4)
        self.assertEqual(get_deletions(None), all_deletions)

        deletions = get_deletions(
            [
                dict(
                    id="os1",
                    start_in_beats_unfolded=start,
                    end_in_beats_unfolded=end,
                    start_in_beats_original=start,
                    end_in_beats_original=end,
                    section_attr_list=[],
                )
            ]
        )
        self.assertEqual(all_deletions - deletions, set(omitted))
        self.assertTrue(set(at_end) <= deletions)

    def test_virtual_note_ids(self):
        """
        Test that virtual notes have the same IDs as the notes they point to, and
        that attributes of the score are not repeated.
        """
        score = load_score(MOZART_VARIATION_FILES["musicxml"])
        sna = score.note_array()
        ppart = PerformedPart(
            notes=[
                dict(id=i, midi_pitch=60, note_on=i, note_off=i + 0.5, velocity=64)
                for i in range(2)
            ]
        )
        mf = matchfile_from_alignment(
            alignment=[
                dict(label="match", score_id=sna["id"][0], performance_id=0),
                dict(
                    label="match",
                    score_id=sna["id"][1],
                    performance_id=1,
                    score_attributes_list=["v1", "attribute"],
                ),
                dict(label="match", score_id=sna["id"][2], performance_id=0),
                dict(label="match", score_id=sna["id"][0], performance_id=1),
            ],
            ppart=ppart,
            spart=score[0],
        )
        self.assertEqual(
            sorted((sn.Anchor, n.Id) for sn, n in mf.note_pairs),
            sorted(
                [
                    (sna["id"][0], "n0"),
                    (sna["id"][1], "n1"),
                    (sna["id"][2], "n0"),
                    (sna["id"][0], "n1"),
                ]
            ),
        )
        snote = next(sn for sn in mf.snotes if sn.Anchor == sna["id"][1])
        self.assertEqual(snote.ScoreAttributesList.count("v1"), 1)
        self.assertEqual(snote.ScoreAttributesList.count("attribute"), 1)
