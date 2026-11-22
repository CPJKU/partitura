"""Tests for MusicXML fingering annotations that cannot be read as integers."""

from io import BytesIO
import unittest

from lxml import etree

from partitura import load_musicxml
from partitura.io.importmusicxml import parse_fingering

_SCORE_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>1</divisions></attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>
        <notations><technical><fingering>1</fingering></technical></notations>
      </note>
      <note>
        <pitch><step>D</step><octave>4</octave></pitch><duration>1</duration>
        <notations><technical>
          <fingering placement="above">{}</fingering>
        </technical></notations>
      </note>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch><duration>1</duration>
        <notations><technical><fingering>4_1</fingering></technical></notations>
      </note>
    </measure>
  </part>
</score-partwise>
"""


class TestFingering(unittest.TestCase):
    def test_score_with_unparseable_fingering(self):
        for text in ("simile", ""):
            with self.subTest(text=text):
                xml = BytesIO(_SCORE_XML.format(text).encode("utf-8"))
                with self.assertWarnsRegex(UserWarning, "Cannot parse fingering info"):
                    part = load_musicxml(xml)[0]

                self.assertEqual([n.midi_pitch for n in part.notes], [60, 62, 64])
                self.assertEqual([n.duration for n in part.notes], [1, 1, 1])
                self.assertEqual(
                    [n.technical[0].fingering for n in part.notes], [1, None, 4]
                )
                self.assertEqual(part.notes[1].technical[0].placement, "above")

    def test_unparseable_fingering(self):
        for text in ("simile", " ", None):
            with self.subTest(text=text):
                element = etree.Element("fingering", placement="above")
                element.text = text
                with self.assertWarnsRegex(UserWarning, "Cannot parse fingering info"):
                    fingering = parse_fingering(element)
                self.assertIsNone(fingering.fingering)
                self.assertIsNone(fingering.info)
                self.assertEqual(fingering.placement, "above")

    def test_numeric_fingerings(self):
        for text, expected in (("1", 1), ("4_1", 4), ("2 3", 2)):
            with self.subTest(text=text):
                element = etree.Element("fingering")
                element.text = text
                self.assertEqual(parse_fingering(element).fingering, expected)


if __name__ == "__main__":
    unittest.main()
