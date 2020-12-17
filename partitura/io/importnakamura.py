#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for parsing score-to-performance alignments
in Nakamura et al.'s [1]_ format.

References
----------
.. [1] Nakamura, E., Yoshii, K. and Katayose, H. (2017) "Performance Error
       Detection and Post-Processing for Fast and Accurate Symbolic Music
       Alignment"
"""
from collections import defaultdict
from fractions import Fraction
import logging
from operator import attrgetter, itemgetter
import re
import warnings

import numpy as np
from scipy.interpolate import interp1d

from partitura.io.exportmatch import matchfile_from_alignment
from partitura.io.importmidi import load_performance_midi
from partitura.io.importmusicxml import load_musicxml
from partitura.performance import PerformedPart
from partitura.score import PartGroup
import partitura.score as score
from partitura.utils import (
    pitch_spelling_to_midi_pitch,
    ensure_pitch_spelling_format,
    ALTER_SIGNS,
    key_name_to_fifths_mode,
    iter_current_next,
    partition,
    estimate_clef_properties,
    match_note_arrays
)


class MatchError(Exception):
    pass

class NakamuraMatchLine(object):
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

def parse_nakamuramatchline(l):
    """
    Return objects representing the line as line or comment

    Parameters
    ----------
    l : str
        Line of the match file

    Returns
    -------
    nakamuramatchline : subclass of `NakamuraMatchLine`
       Object representing the line.
    """

    from_NakamuraMatchLine_methods = [NakamuraMatchLine.from_line]
    nakamuramatchline = False
    for from_NakamuraMatchLine in from_NakamuraMatchLine_methods:
        try:
            nakamuramatchline = from_NakamuraMatchLine(l)
            break
        except MatchError:
            continue

    return nakamuramatchline


class NakamuraMatchFile(object):
    """
    Class for representing Nakamura et al.'s match.txt Files
    """

    def __init__(self, filename):
        self.name = filename

        with open(filename) as f:
            self.lines = np.array([parse_nakamuramatchline(l) for l in f.read().splitlines()])

    def iter_notes(self):
        """
        Iterate over all performed notes
        """
        for x in self.lines:
            if x is None:
                continue
            else:
                yield x



def load_nakamuramatch(fn, pedal_threshold=64, first_note_at_zero=False):
    """ Load a match file as returned by Nakamura et al.'s MIDI to musicxml alignment
    
    Fields of the file format as specified here https://midialignment.github.io/MANUAL.pdf
    ID (onset time) (offset time) (spelled pitch) (onset velocity)(offset velocity) channel (match status) (score time) (note ID)(error index) (skip index)

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
    mf = NakamuraMatchFile(fn)

    ######## Generate PerformedPart #########
    ppart = performed_part_from_nakamuramatch(mf, pedal_threshold, first_note_at_zero)

    ###### Alignment ########
    alignment = alignment_from_nakamuramatch(mf)

    return mf, ppart, alignment

def alignment_from_nakamuramatch(mf):
    result = []
    for l in mf.iter_notes():
        result.append(dict(label='match',
                            score_id=l.snoteID,
                            performance_id=l.number))
    return result

def performed_part_from_nakamuramatch(mf, pedal_threshold=64, first_note_at_zero=False):
    """Make PerformedPart from performance info in a match.txt file

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
    notes = []

    first_note = next(mf.iter_notes(), None)
    if first_note and first_note_at_zero:
        offset = first_note.onset
    else:
        offset = 0

    for note in mf.iter_notes():
        notes.append(dict(id=note.number,
                          midi_pitch=note_name_to_midi_pitch(note.notename),
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



class NakamuraCorrespLine(object):
    field_names = ["alignID", "alignOntime", "alignSitch",
                    "alignPitch", "alignOnvel", "refID",
                    "refOntime", "refSitch", "refPitch", "refOnvel"]
    
    out_pattern = '{alignID}\t{alignOntime}\t{alignSitch}'\
                      '\t{alignPitch}\t{alignOnvel}'\
                      '\t{refID}\t{refOntime}'\
                      '\t{refSitch}\t{refPitch}\t{refOnvel}'

    def __init__(self, alignID, alignOntime, alignSitch,
                    alignPitch, alignOnvel, refID,
                    refOntime, refSitch, refPitch, refOnvel):
        
        self.id0 = str(alignID)
        self.onset0 = float(alignOntime)
        self.pitch0 = int(alignPitch)
        self.alignSitch = str(alignSitch)
        self.alignOnvel = int(alignOnvel)

        self.id1 = str(refID)
        self.onset1 = float(refOntime)
        self.pitch1 = int(refPitch)
        self.refSitch = str(refSitch)
        self.refOnvel = int(refOnvel)

    @classmethod
    def from_line(cls, correspline, pos=0):
        line_split = correspline.split("\t")
        del line_split[-1]

        if len(line_split) != 10:
            return None
        else:
            kwargs = dict(zip(cls.field_names, line_split))
            corresp_line = cls(**kwargs)
            return corresp_line
    
    @property
    def corresp_line(self):
        self.out_pattern.format(
        alignID = self.id0,
        alignOntime = self.onset0,
        alignPitch = self.pitch0,
        alignSitch = self.alignSitch,
        alignOnvel = self.alignOnvel,
        refID = self.id1,
        refOntime = self.onset1,
        refPitch = self.pitch1,
        refSitch = self.refSitch,
        refOnvel = self.refOnvel
        )
        

def parse_nakamuracorrespline(l):
    """
    Return objects representing the line as line or comment

    Parameters
    ----------
    l : str
        Line of the match file

    Returns
    -------
    nakamuracorrespline : `NakamuraCorrespLine`
       Object representing the line.
    """

    from_NakamuraCorrespLine_methods = [NakamuraCorrespLine.from_line]
    nakamuracorrespline = None
    for from_NakamuraCorrespLine in from_NakamuraCorrespLine_methods:
        try:
            nakamuracorrespline = from_NakamuraCorrespLine(l)
            break
        except MatchError:
            continue

    return nakamuracorrespline


class NakamuraCorrespFile(object):
    """
    Class for representing nakamura's corresp.txt Files
    """

    def __init__(self, filename):
        self.name = filename

        with open(filename) as f:
            self.lines = np.array([parse_nakamuracorrespline(l) for l in f.read().splitlines()])

    @property
    def note_pairs(self):
        raise NotImplementedError

    @property
    def notes(self):
        raise NotImplementedError

    def iter_notes(self):
        """
        Iterate over all note pairs
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

    @property
    def note_arrays(self):
        """
        generate a tuple of (performance-, score-) note arrays with pitch, id and onset information (in seconds).

        Returns
        -------
        (array_performance, array_score) : tuple
            a tuple of structured arrays

        """
        fields = [('onset', 'f4'),
                  ('pitch', 'i4'),
                  ('id', 'U256')]

        note_array0 = []
        note_array1 = []
        for l in self.iter_notes():
            if l.id0 != "*":
                note_array0.append((l.onset0, l.pitch0, l.id0))
            if l.id1 != "*":
                note_array1.append((l.onset1, l.pitch1, l.id1))
        ar0 = np.array(note_array0, dtype=fields)
        ar1 = np.array(note_array1, dtype=fields)
        return ar0, ar1


    @property
    def alignment(self):
        result = []
        for l in self.iter_notes():
            if l.id0 == "*":
                result.append(dict(label='deletion',
                                   performance_id=l.id0,
                                score_id=l.id1))
            elif l.id1 == "*":
                result.append(dict(label='insertion',
                                performance_id=int(l.id0),
                                score_id=l.id1))
            else:
                result.append(dict(label='match',
                                performance_id=int(l.id0),
                                score_id=l.id1))
        return result



def load_nakamuracorresp(fn, pedal_threshold=64, first_note_at_zero=False):
    """ Load a corresp file as returned by Nakamura et al.'s MIDI to MIDI alignment.

    Fields of the file format as specified here: https://midialignment.github.io/MANUAL.pdf
    (ID) (onset time) (spelled pitch) (integer pitch) (onset velocity)

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
    array_performance : structured array
        structured array of performed notes
    array_score : structured array
        structured array of score notes
    alignment : list
        The score--performance alignment, a list of dictionaries
    """
    cf = NakamuraCorrespFile(fn)
    array_performance, array_score = cf.note_arrays
    alignment = cf.alignment
    return array_performance, array_score, alignment


def note_name_to_midi_pitch(notename):
    """
    Utility function to convert the Nakamura pitch spelling to MIDI pitches
    """
    pitch = 0
    keys = ["A","B","C","D","E","F","G","\#","b","0","1","2","3","4","5","6","7","8", "x", "bb"]
    values = [21,23,12,14,16,17,19,1,-1,0,12,24,36,48,60,72,84,96,2,-1]
    for k, v in zip(keys, values):
        if k in notename:
            pitch += v
    return pitch
