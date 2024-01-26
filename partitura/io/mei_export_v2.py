#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for exporting MEI files.
"""
import math
from collections import defaultdict
from lxml import etree
import partitura.score as spt
from operator import itemgetter
from typing import Optional
from partitura.utils import partition, iter_current_next, to_quarter_tempo
import numpy as np
from partitura.utils.misc import deprecated_alias, PathLike

__all__ = ["save_mei"]

ALTER_TO_MEI = {
    -2: "ff",
    -1: "f",
    0: "n",
    1: "s",
    2: "ss",
}

DOCTYPE = '<?xml version="1.0" encoding="UTF-8"?> /n <?xml-model href="https://music-encoding.org/schema/4.0.1/mei-CMN.rng" type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?> \n <?xml-model href="https://music-encoding.org/schema/4.0.1/mei-CMN.rng" type="application/xml" schematypens="http://purl.oclc.org/dsdl/schematron"?>'

class MEIExporter:
    def __init__(self, part):
        self.part = part
        self.element_counter = 0

    def elc_id(self):
        # transforms an integer number to 8-digit string
        # The number is right aligned and padded with zeros
        return str(self.element_counter).zfill(10)

    def export_to_mei(self):
        # Create root MEI element
        mei = etree.Element('mei')

        # Create child elements
        mei_head = etree.SubElement(mei, 'meiHead')
        file_desc = etree.SubElement(mei_head, 'fileDesc')
        music = etree.SubElement(mei, 'music')
        body = etree.SubElement(music, 'body')
        mdiv = etree.SubElement(body, 'mdiv')
        score = etree.SubElement(mdiv, 'score')
        score_def = etree.SubElement(score, 'scoreDef')
        staff_grp = etree.SubElement(score_def, 'staffGrp')
        staff_def = etree.SubElement(staff_grp, 'staffDef')
        section = etree.SubElement(score, 'section')

        # Iterate over part's timeline
        for measure in self.part.measures:
            # Create measure element
            xml_el = etree.SubElement(section, 'measure')
            self._handle_measure(measure, xml_el)

        return mei

    def _handle_measure(self, measure, xml_el):
        # Add measure number
        xml_el.set('n', str(measure.number))
        xml_el.set('id', "measure-" + self.elc_id())
        note_or_rest_elements = np.array(list(self.part.iter_all(spt.GenericNote, start=measure.start.t, end=measure.end.t, include_subclasses=True)))
        # Separate by staff
        staffs = np.vectorize(lambda x: x.staff)(note_or_rest_elements)
        unique_staffs, staff_inverse_map = np.unique(staffs, return_inverse=True)
        for i, staff in enumerate(unique_staffs):
            staff_el = etree.SubElement(xml_el, 'staff')
            # Add staff number
            staff_el.set('n', str(staff))
            staff_el.set('id', "staff-" + self.elc_id())
            staff_notes = note_or_rest_elements[staff_inverse_map == i]
            # Separate by voice
            voices = np.vectorize(lambda x: x.voice)(staff_notes)
            unique_voices, voice_inverse_map = np.unique(voices, return_inverse=True)
            for j, voice in enumerate(unique_voices):
                voice_el = etree.SubElement(staff_el, 'layer')
                voice_el.set('n', str(voice))
                voice_el.set('id', "voice-" + self.elc_id())
                voice_notes = staff_notes[voice_inverse_map == j]
                # Sort by onset
                voice_notes = sorted(voice_notes, key=lambda x: x.start.t)
                # group by start time
                for _, group in iter_current_next(voice_notes, key=lambda x: x.start.t):
                    if len(group) == 1:
                        self._handle_note_or_rest(group[0], voice_el)
                    else:
                        self._handle_chord(group, voice_el)

        return xml_el

    def _handle_chord(self, chord, xml_voice_el):
        chord_el = etree.SubElement(xml_voice_el, 'chord')
        chord_el.set('id', "chord-" + self.elc_id())
        for note in chord:
            self._handle_note_or_rest(note, chord_el)

    def _handle_note_or_rest(self, note, xml_voice_el):
        if isinstance(note, spt.Rest):
            self._handle_rest(note, xml_voice_el)
        else:
            self._handle_note(note, xml_voice_el)

    def _handle_rest(self, rest, xml_voice_el):
        rest_el = etree.SubElement(xml_voice_el, 'rest')
        rest_el.set('dur', str(rest.duration))
        rest_el.set('id', "rest-" + self.elc_id())

    def _handle_note(self, note, xml_voice_el):
        note_el = etree.SubElement(xml_voice_el, 'note')
        note_el.set('dur', str(note.duration))
        note_el.set('id', "note-" + self.elc_id())
        note_el.set('oct', str(note.octave))
        note_el.set('pname', note.step.lower())
        if note.alter is not None:
            accidental = etree.SubElement(note_el, 'accid')
            accidental.set('id', "accid-" + self.elc_id())
            accidental.set('accid', ALTER_TO_MEI[note.alter])


@deprecated_alias(parts="score_data")
def save_mei(
    score_data: spt.ScoreLike,
    out: Optional[PathLike] = None,
) -> Optional[str]:
    """
    Save a one or more Part or PartGroup instances in MEI format.

    Parameters
    ----------
    score_data : Score, list, Part, or PartGroup
        The musical score to be saved. A :class:`partitura.score.Score` object,
        a :class:`partitura.score.Part`, a :class:`partitura.score.PartGroup` or
        a list of these.
    out: str, file-like object, or None, optional
        Output file

    Returns
    -------
    None or str
        If no output file is specified using `out` the function returns the
        MEI data as a string. Otherwise the function returns None.
    """

    if isinstance(score_data, spt.Score):
        score_data = spt.merge_parts(score_data.parts)

    exporter = MEIExporter(score_data)
    root = exporter.export_to_mei()

    if out is None:
        return etree.tostring(
            root.getroottree(),
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=True,
            doctype=DOCTYPE,
        )
