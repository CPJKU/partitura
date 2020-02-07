#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for parsing nakamura matchfiles
"""
import re
from fractions import Fraction
from collections import defaultdict
from operator import attrgetter, itemgetter
import logging
import warnings

import numpy as np
from scipy.interpolate import interp1d

from partitura.utils import (
    pitch_spelling_to_midi_pitch,
    ensure_pitch_spelling_format,
    ALTER_SIGNS,
    key_name_to_fifths_mode,
    iter_current_next,
    partition,
    estimate_clef_properties,
    notes_to_notearray
)

from partitura.performance import PerformedPart
import partitura.score as score


###################################################
class NakamuraLine(object):
    field_names = ['number', 'onset', 'offset', 'notename',
                    'velocityonset', 'velocityoffset',
                    'channel', 'matchstatus', 'scoretime', 'snoteID',
                    'errorindex', 'skipindex']
    def __init__(self,number, onset, offset, notename,
                    velocityonset, velocityoffset,
                    channel, matchstatus, scoretime, snoteID,
                    errorindex, skipindex):
        self.number = int(number)
        self.onset = float(onset)
        self.offset = float(offset)
        self.velocityonset = int(velocityonset)

        self.notename = notename
        self.scoretime = float(scoretime)
        self.snoteID = snoteID

    @classmethod
    def from_line(cls, matchline, pos=0):
        line_split = matchline.split("\t")
        if len(line_split) != 12:
            return None
        else:
            kwargs = dict(zip(cls.field_names, line_split))
            match_line = cls(**kwargs)
            return match_line

def parse_nakamuraline(l):
    """
    Return objects representing the line as line or comment

    Parameters
    ----------
    l : str
        Line of the match file

    Returns
    -------
    matchline : subclass of `MatchLine`
       Object representing the line.
    """

    from_nakamuraline_methods = [NakamuraLine.from_line]
    nakamuraline = False
    for from_nakamuraline in from_nakamuraline_methods:
        try:
            nakamuraline = from_nakamuraline(l)
            break
        except MatchError:
            continue

    return nakamuraline


class NakamuraFile(object):
    """
    Class for representing nakamura's match.txt Files
    """

    def __init__(self, filename):
        self.name = filename

        with open(filename) as f:
            self.lines = np.array([parse_nakamuraline(l) for l in f.read().splitlines()])

    @property
    def note_pairs(self):
        raise NotImplementedError

    @property
    def notes(self):
        raise NotImplementedError

    def iter_notes(self):
        """
        Iterate over all performed notes
        """
        for x in self.lines:
            if x is None:
                continue
            else:
                yield x

    @property
    def snotes(self):
        raise NotImplementedError


    @property
    def sustain_pedal(self):
        raise NotImplementedError



def load_nakamura(fn, pedal_threshold=64, first_note_at_zero=False):
    """Load a nakamuramatchfile.

    Parameters
    ----------
    fn : str
        The nakamura match.txt-file
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.

    Returns
    -------
    ppart : list
        The performed part, a list of dictionaries
    alignment : list
        The score--performance alignment, a list of dictionaries

    """
    # Parse Matchfile
    mf = NakamuraFile(fn)

    ######## Generate PerformedPart #########

    ppart = performed_part_from_nakamura(mf, pedal_threshold, first_note_at_zero)

    ###### Alignment ########

    alignment = alignment_from_nakamura(mf)

    return mf, ppart, alignment


def alignment_from_nakamura(mf):
    result = []
    for l in mf.iter_notes():
        result.append(dict(label='match',
                            score_id=l.snoteID,
                            performance_id=l.number))
    return result

# PERFORMANCE PART FROM MATCHFILE stuff

def performed_part_from_nakamura(mf, pedal_threshold=64, first_note_at_zero=False):
    """Make PerformedPart from performance info in a Nakamura match.txt file

    Parameters
    ----------
    mf : nakamuramatchfile
        A MatchFile instance
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.

    Returns
    -------
    ppart : PerformedPart
        A performed part

    """
    # PerformedNote instances for all MatchNotes
    notes = []

    first_note = next(mf.iter_notes(), None)
    if first_note and first_note_at_zero:
        offset = first_note.onset
    else:
        offset = 0

    for note in mf.iter_notes():
        notes.append(dict(id=note.number,
                          midi_pitch=NoteName2MidiPitch(note.notename),
                          note_on=note.onset  - offset,
                          note_off=note.offset  - offset,
                          sound_off=note.offset - offset,
                          velocity=note.velocityonset
                          ))

    ppart = PerformedPart(id='P1',
                          part_name="unknown_part_1",
                          notes=notes,
                          sustain_pedal_threshold=pedal_threshold)
    return ppart

def NoteName2MidiPitch(NoteName):
    """
    Utility function to convert the Nakamura pitch spelling to MIDI pitches
    """
    pitch = 0
    keys = ["A","B","C","D","E","F","G","\#","b","0","1","2","3","4","5","6","7","8"]
    values = [21,23,12,14,16,17,19,      1, -1,   0,  12, 24, 36, 48, 60, 72, 84, 96]
    for k, v in zip(keys, values):
        if k in NoteName:
            pitch += v
    return pitch
