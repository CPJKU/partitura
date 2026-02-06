#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains test functions for io of the harmony musicxml tag.
"""

import unittest
import tempfile
import os
import warnings
from partitura import load_musicxml, save_musicxml
from partitura.score import ChordSymbol, RomanNumeral, Part
from tests import HARMONY_TESTFILES


class HarmonyImportTester(unittest.TestCase):
    part = load_musicxml(HARMONY_TESTFILES[0])[0]

    def test_chordsymbol(self):
        roots = list()
        kinds = list()
        for cs in self.part.iter_all(ChordSymbol):
            roots.append(cs.root)
            kinds.append(cs.kind)
        self.assertEqual(roots, ["C", "G"])
        self.assertEqual(kinds, ["m", "7"])

    def test_romanNumeral(self):
        text = list()
        for cs in self.part.iter_all(RomanNumeral):
            text.append(cs.text)
        self.assertEqual(text, ["i", "V7"])


class TestChordSymbol(unittest.TestCase):
    """Unit tests for the ChordSymbol class."""

    def test_chordsymbol_basic(self):
        """Test basic ChordSymbol creation with root and kind."""
        cs = ChordSymbol(root="C", kind="major")
        self.assertEqual(cs.root, "C")
        self.assertEqual(cs.kind, "major")
        self.assertEqual(cs.alter, 0)
        self.assertIsNone(cs.bass)

    def test_chordsymbol_sharp(self):
        """Test ChordSymbol with sharp (alter=1)."""
        cs = ChordSymbol(root="F", kind="major", alter=1)
        self.assertEqual(cs.root, "F")
        self.assertEqual(cs.alter, 1)
        self.assertIn("F#", cs.text)

    def test_chordsymbol_flat(self):
        """Test ChordSymbol with flat (alter=-1)."""
        cs = ChordSymbol(root="B", kind="minor", alter=-1)
        self.assertEqual(cs.root, "B")
        self.assertEqual(cs.alter, -1)
        self.assertIn("Bb", cs.text)

    def test_chordsymbol_double_sharp(self):
        """Test ChordSymbol with double sharp (alter=2)."""
        cs = ChordSymbol(root="C", kind="major", alter=2)
        self.assertEqual(cs.alter, 2)
        self.assertIn("C##", cs.text)

    def test_chordsymbol_double_flat(self):
        """Test ChordSymbol with double flat (alter=-2)."""
        cs = ChordSymbol(root="D", kind="minor", alter=-2)
        self.assertEqual(cs.alter, -2)
        self.assertIn("Dbb", cs.text)

    def test_chordsymbol_no_kind(self):
        """Test ChordSymbol when kind is None."""
        cs = ChordSymbol(root="C", kind=None)
        self.assertEqual(cs.root, "C")
        self.assertIsNone(cs.kind)
        # Text should not have "/None" or similar
        self.assertEqual(cs.text, "C")

    def test_chordsymbol_with_bass(self):
        """Test ChordSymbol with bass note."""
        cs = ChordSymbol(root="C", kind="major", bass="G")
        self.assertEqual(cs.bass, "G")
        self.assertIn("/G", cs.text)

    def test_chordsymbol_text_property(self):
        """Verify text property is correctly constructed as root+alter/kind/bass."""
        cs = ChordSymbol(root="F", kind="7", bass="C", alter=1)
        # Expected: "F#/7/C"
        self.assertEqual(cs.text, "F#/7/C")

    def test_chordsymbol_text_no_alter(self):
        """Verify text property with no alteration."""
        cs = ChordSymbol(root="G", kind="m")
        self.assertEqual(cs.text, "G/m")

    def test_chordsymbol_alter_zero_no_symbol(self):
        """Verify alter=0 produces no accidental symbol in text."""
        cs = ChordSymbol(root="C", kind="major", alter=0)
        self.assertEqual(cs.text, "C/major")
        self.assertNotIn("#", cs.text)
        self.assertNotIn("b", cs.text)

    def test_chordsymbol_str(self):
        """Verify __str__ format includes root/alter/kind."""
        cs = ChordSymbol(root="D", kind="dim", alter=-1)
        s = str(cs)
        self.assertIn("D/-1/dim", s)


class TestChordSymbolImport(unittest.TestCase):
    """Tests for importing ChordSymbols from MusicXML."""

    def _make_musicxml(self, harmony_xml):
        """Helper to wrap harmony XML in a minimal MusicXML document."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Test</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <clef>
          <sign>G</sign>
          <line>2</line>
        </clef>
      </attributes>
      {harmony_xml}
      <note>
        <pitch>
          <step>C</step>
          <octave>4</octave>
        </pitch>
        <duration>4</duration>
        <type>whole</type>
      </note>
    </measure>
  </part>
</score-partwise>"""

    def _load_from_string(self, xml_string):
        """Load MusicXML from a string using a temp file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".musicxml", delete=False
        ) as f:
            f.write(xml_string)
            f.flush()
            try:
                return load_musicxml(f.name)
            finally:
                os.unlink(f.name)

    def test_import_chord_with_root_alter_sharp(self):
        """Test importing chord with <root-alter>1</root-alter> (sharp)."""
        harmony = """
        <harmony>
          <root>
            <root-step>F</root-step>
            <root-alter>1</root-alter>
          </root>
          <kind text="maj">major</kind>
        </harmony>"""
        parts = self._load_from_string(self._make_musicxml(harmony))
        chords = list(parts[0].iter_all(ChordSymbol))
        self.assertEqual(len(chords), 1)
        self.assertEqual(chords[0].root, "F")
        self.assertEqual(chords[0].alter, 1)
        self.assertEqual(chords[0].kind, "maj")

    def test_import_chord_with_root_alter_flat(self):
        """Test importing chord with <root-alter>-1</root-alter> (flat)."""
        harmony = """
        <harmony>
          <root>
            <root-step>B</root-step>
            <root-alter>-1</root-alter>
          </root>
          <kind text="m">minor</kind>
        </harmony>"""
        parts = self._load_from_string(self._make_musicxml(harmony))
        chords = list(parts[0].iter_all(ChordSymbol))
        self.assertEqual(len(chords), 1)
        self.assertEqual(chords[0].root, "B")
        self.assertEqual(chords[0].alter, -1)
        self.assertEqual(chords[0].kind, "m")

    def test_import_chord_without_root_alter(self):
        """Test importing chord without <root-alter> defaults to 0."""
        harmony = """
        <harmony>
          <root>
            <root-step>C</root-step>
          </root>
          <kind text="7">dominant</kind>
        </harmony>"""
        parts = self._load_from_string(self._make_musicxml(harmony))
        chords = list(parts[0].iter_all(ChordSymbol))
        self.assertEqual(len(chords), 1)
        self.assertEqual(chords[0].root, "C")
        self.assertEqual(chords[0].alter, 0)

    def test_import_chord_various_kinds(self):
        """Test importing chords with various kind values."""
        # Test representative subset: major, minor, dominant, diminished, augmented
        test_cases = [
            ("major", "maj"),
            ("minor", "m"),
            ("dominant", "7"),
            ("diminished", "dim"),
            ("augmented", "aug"),
        ]
        for kind_content, kind_text in test_cases:
            harmony = f"""
            <harmony>
              <root>
                <root-step>C</root-step>
              </root>
              <kind text="{kind_text}">{kind_content}</kind>
            </harmony>"""
            with self.subTest(kind=kind_content):
                parts = self._load_from_string(self._make_musicxml(harmony))
                chords = list(parts[0].iter_all(ChordSymbol))
                self.assertEqual(len(chords), 1)
                self.assertEqual(chords[0].kind, kind_text)

    def test_import_chord_position(self):
        """Test ChordSymbol is placed at correct beat position."""
        # Chord at the beginning (position 0)
        harmony = """
        <harmony>
          <root>
            <root-step>G</root-step>
          </root>
          <kind text="m7">minor-seventh</kind>
        </harmony>"""
        parts = self._load_from_string(self._make_musicxml(harmony))
        chords = list(parts[0].iter_all(ChordSymbol))
        self.assertEqual(len(chords), 1)
        self.assertEqual(chords[0].start.t, 0)

    def test_import_kind_uses_text_attribute(self):
        """Test that kind is extracted from the 'text' attribute, not element content.

        This documents the current backward-compatible behavior where the kind
        value comes from the 'text' attribute of the <kind> element, not its
        text content. For example:
            <kind text="m">minor</kind>
        Results in kind="m", not kind="minor".
        """
        harmony = """
        <harmony>
          <root>
            <root-step>A</root-step>
          </root>
          <kind text="m7b5">half-diminished</kind>
        </harmony>"""
        parts = self._load_from_string(self._make_musicxml(harmony))
        chords = list(parts[0].iter_all(ChordSymbol))
        self.assertEqual(len(chords), 1)
        # Kind should be "m7b5" (text attribute), not "half-diminished" (element content)
        self.assertEqual(chords[0].kind, "m7b5")

    def test_import_kind_without_text_attribute(self):
        """Test importing chord where kind has no text attribute."""
        harmony = """
        <harmony>
          <root>
            <root-step>C</root-step>
          </root>
          <kind>major</kind>
        </harmony>"""
        parts = self._load_from_string(self._make_musicxml(harmony))
        chords = list(parts[0].iter_all(ChordSymbol))
        self.assertEqual(len(chords), 1)
        # When text attribute is missing, kind should be None
        self.assertIsNone(chords[0].kind)

    def test_import_empty_harmony_warning(self):
        """Test that empty harmony tag produces a warning."""
        harmony = """
        <harmony>
        </harmony>"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._load_from_string(self._make_musicxml(harmony))
            # Check that a warning was issued about empty harmony
            harmony_warnings = [
                x for x in w if "empty <harmony>" in str(x.message).lower()
            ]
            self.assertEqual(len(harmony_warnings), 1)


class TestChordSymbolRoundTrip(unittest.TestCase):
    """Tests for export and re-import of ChordSymbols."""

    def _make_part_with_chord(self, root, kind, alter=0, bass=None):
        """Create a Part with a single ChordSymbol and minimal content for export."""
        from partitura.score import Note, TimeSignature, add_measures

        part = Part("P1", "Test Part")
        part.set_quarter_duration(0, 1)

        ts = TimeSignature(4, 4)
        part.add(ts, 0)

        note = Note(step="C", octave=4, voice=1, staff=1)
        part.add(note, 0, 4)

        cs = ChordSymbol(root=root, kind=kind, alter=alter, bass=bass)
        part.add(cs, 0)

        add_measures(part)
        return part

    def _roundtrip(self, part):
        """Export part to MusicXML and re-import it."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".musicxml", delete=False
        ) as f:
            try:
                save_musicxml(part, f.name)
                return load_musicxml(f.name)
            finally:
                os.unlink(f.name)

    def test_roundtrip_chord_basic(self):
        """Test basic chord survives export/re-import."""
        part = self._make_part_with_chord("C", "maj")
        reimported = self._roundtrip(part)
        chords = list(reimported[0].iter_all(ChordSymbol))
        self.assertEqual(len(chords), 1)
        self.assertEqual(chords[0].root, "C")
        self.assertEqual(chords[0].kind, "maj")

    def test_roundtrip_chord_with_sharp(self):
        """Test chord with sharp alter survives round-trip."""
        part = self._make_part_with_chord("F", "maj", alter=1)
        reimported = self._roundtrip(part)
        chords = list(reimported[0].iter_all(ChordSymbol))
        self.assertEqual(len(chords), 1)
        self.assertEqual(chords[0].root, "F")
        self.assertEqual(chords[0].alter, 1)
        self.assertEqual(chords[0].kind, "maj")

    def test_roundtrip_chord_with_flat(self):
        """Test chord with flat alter survives round-trip."""
        part = self._make_part_with_chord("B", "m", alter=-1)
        reimported = self._roundtrip(part)
        chords = list(reimported[0].iter_all(ChordSymbol))
        self.assertEqual(len(chords), 1)
        self.assertEqual(chords[0].root, "B")
        self.assertEqual(chords[0].alter, -1)
        self.assertEqual(chords[0].kind, "m")

    @unittest.skip("Bass import not yet implemented in importmusicxml._handle_harmony")
    def test_roundtrip_chord_with_bass(self):
        """Test chord with bass note survives round-trip.

        NOTE: Currently skipped because bass import is not implemented.
        Export handles bass correctly, but import ignores <bass> element.
        """
        part = self._make_part_with_chord("C", "maj", bass="G")
        reimported = self._roundtrip(part)
        chords = list(reimported[0].iter_all(ChordSymbol))
        self.assertEqual(len(chords), 1)
        self.assertEqual(chords[0].root, "C")
        self.assertEqual(chords[0].bass, "G")

    def test_roundtrip_chord_no_kind(self):
        """Test chord with kind=None survives round-trip."""
        part = self._make_part_with_chord("D", None)
        reimported = self._roundtrip(part)
        chords = list(reimported[0].iter_all(ChordSymbol))
        self.assertEqual(len(chords), 1)
        self.assertEqual(chords[0].root, "D")
        # kind will be None or empty string after round-trip
        self.assertIn(chords[0].kind, [None, ""])
