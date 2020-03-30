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


class MatchError(Exception):
    pass

###################################################
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
    Class for representing nakamura's match.txt Files
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
    """ Load a nakamuramatchfile.

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

# PERFORMANCE PART FROM MATCHFILE stuff
def performed_part_from_nakamuramatch(mf, pedal_threshold=64, first_note_at_zero=False):
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









# NAKAMURA CORRESP FILES FROM MIDI TO MIDI ALIGNMENT
class NakamuraCorrespLine(object):
    field_names = ["alignID", "alignOntime", "alignSitch",
                    "alignPitch", "alignOnvel", "refID",
                    "refOntime", "refSitch", "refPitch", "refOnvel"]
    def __init__(self, alignID, alignOntime, alignSitch,
                    alignPitch, alignOnvel, refID,
                    refOntime, refSitch, refPitch, refOnvel):

        self.id0 = str(alignID)
        self.onset0 = float(alignOntime)
        self.pitch0 = int(alignPitch)

        self.id1 = str(refID)
        self.onset1 = float(refOntime)
        self.pitch1 = int(refPitch)

    @classmethod
    def from_line(cls, correspline, pos=0):
        line_split = correspline.split("\t")
        del line_split[-1]

        if len(line_split) != 10:
            return None
        else:
            kwargs = dict(zip(cls.field_names, line_split))
            match_line = cls(**kwargs)
            return match_line

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
    Class for representing nakamura's match.txt Files
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
            # if l.id0 == "*":
            #     print("performance omission at score id: ", l.id1)
            # if l.id1 == "*":
            #     print("performance insertion at performance id: ", l.id0)
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
    """ Load a nakamuramatchfile.

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
    # Parse Matchfile
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
    values = [21,23,12,14,16,17,19,      1, -1,   0,  12, 24, 36, 48, 60, 72, 84, 96, 2, -1 ]#the last one is only -1 since the single b is also found, adding up to -2
    for k, v in zip(keys, values):
        if k in notename:
            pitch += v
    return pitch


def match_by_pitch_and_position(dicts_0, dicts_1,
                                name_pitch_0 = "pitch", name_pitch_1 = "pitch",
                                name_position_0 = "onset", name_position_1 = "onset",
                                name_id_0 = "id", name_id_1 = "id",
                                position_alignment_threshold = 0.005):
    """
    Utility function to align two note_arrays or lists of note dictionaries by pitch and score position
    """

    alignment = []
    for ele_0 in dicts_0:
        dict = {"id_0": [ele_0[name_id_0]], "id_1":[]}
        for ele_1 in dicts_1:
            if np.abs(ele_0[name_position_0]-ele_1[name_position_1]) < position_alignment_threshold and ele_0[name_pitch_0] == ele_1[name_pitch_1]:
                dict["id_1"].append(ele_1[name_id_1])

        if len(dict["id_1"]) > 1:
            print(dict["id_0"], " of dicts_0 has multiple matches in dicts_1: ", dict["id_1"])
        if len(dict["id_1"]) < 1:
            print(dict["id_0"], " of dicts_0 has no matches in dicts_1")

        alignment.append(dict)

    return alignment










from partitura.utils import match_note_arrays
from partitura.io.importmusicxml import load_musicxml
from partitura.io.importmidi import load_performance_midi
from partitura.io.exportmatch import matchfile_from_alignment
from partitura.score import PartGroup

def alignment_from_corresp_pipeline(corresp_file,
                                    performance_midi,
                                    score_midi,
                                    musicxml):
    """
    this function aligns a performance midi and a musicxml via a given score midi and a nakamura corresp file.
    This code changes the corresp file alignment in place to an xml (score part) <-> performed midi (performance part).
    The matching pipeline looks as follows: xml <-(by quarters)-> score midi <-(by seconds)->
                                            score midi list in corresp <-(nakamura alignment func)->
                                            performed midi list in corresp <-(by seconds)->
                                            performed midi
    Parameters
    ----------
    corresp_file : str
        filename of the corresp file
    performance_midi : str
        filename of the performance midi file
    score_midi : str
        filename of the score midi file
    musicxml : str
        filename of the musicxml file

    Returns
    -------
    alignment : list
        The score--performance alignment, a list of dictionaries
    ppartp_midi : partitura.performance.PerformedPart
        An instance of `PerformedPart` containing performance information.
    part_musicxml : partitura.score.Part
        An instance of `Part` containing score information.
    (length_in_sec, length_in_quarter, average_tempo_sec_per_quarter) : tuple
        Three float values indicating the total length of the performance in seconds, the total length of the score in quartes, and the average speed in seconds/quarter

    """

    """

    part_musicxml = load_musicxml(musicxml, force_note_ids=True)
    if isinstance(part_musicxml, PartGroup):
        print("LIST OF PARTS (PARTGROUP.CHILDREN): ", part_musicxml.children)

        score_xml_note_array0 = [part.note_array for part in part_musicxml.children]
        score_xml_note_array = np.concatenate(score_xml_note_array0)

        if len(part_musicxml.children) > 1:

            raise ValueError('MANY PARTS IN PARTGROUP; BEWARE!')
        part_musicxml = part_musicxml.children[0]

    elif isinstance(part_musicxml, list):
        print("LIST OF PARTS: ", part_musicxml)
        raise ValueError('MANY PARTS IN LIST; BEWARE!')
    else:
        score_xml_note_array = part_musicxml.note_array
    # subtract anacrusis to start from zero
    score_xml_note_array["onset"] -= score_xml_note_array["onset"].min()

    ppart_midi_quart = load_performance_midi(score_midi, merge_tracks=True, time_in_quarter=True)
    score_midi_note_array = ppart_midi_quart.note_array

    ppart_midi_sec = load_performance_midi(score_midi, merge_tracks=True)
    perf_midi_note_array = ppart_midi_sec.note_array

    ppartp_midi = load_performance_midi(performance_midi, merge_tracks=True)
    ppart_midi_note_array = ppartp_midi.note_array

    array_performance, array_score, alignment = load_nakamuracorresp(corresp_file)

    print("part in quarters from xml, last onset in quarters: ", score_xml_note_array["onset"].max(), len(score_xml_note_array))
    print("ppart in quarters from score midi, last onset in quarters: ", score_midi_note_array["p_onset"].max(), len(score_midi_note_array))
    print("ppart in seconds from score midi, last onset in seconds: ", perf_midi_note_array["p_onset"].max(), len(perf_midi_note_array))
    print("ppart performance from midi", len(ppart_midi_note_array))
    print("note arrays and alignment from nakamura corresp, last (score) onset in seconds: ", array_score["onset"].max(), len(array_score))

    # import pdb; pdb.set_trace()
    # performance metadata:
    length_in_sec = array_performance["onset"].max()-array_performance["onset"].min()
    length_in_quarter = score_midi_note_array["p_onset"].max()-score_midi_note_array["p_onset"].min()
    average_tempo_sec_per_quarter = length_in_sec/length_in_quarter


    # original_fields = [('id', '<U256'), ('pitch', '<i4'), ('p_onset', '<f4'), ('p_duration', '<f4'), ('velocity', '<i4')]
    new_fields = [('id', '<U256'), ('pitch', '<i4'), ('onset', '<f4'), ('duration', '<f4'), ('velocity', '<i4')]

    score_midi_note_array.dtype = new_fields
    perf_midi_note_array.dtype = new_fields
    ppart_midi_note_array.dtype = new_fields


    match_quarter, match_quarter_note = match_note_arrays(score_xml_note_array, score_midi_note_array,
                      array_type='score', epsilon=0.01,
                      first_note_at_zero=True,
                      check_duration=False,
                      return_note_idxs=True)

    # a small epsilon gives good results but misses all the acciaccaturas and broken chords, etc
    # take the missed notes and do it again, more generously:
    keys_not_matched_xml =[i for i in range(len(score_xml_note_array)) if i not in match_quarter[:,0]]
    keys_not_matched_midi =[i for i in range(len(score_midi_note_array)) if i not in match_quarter[:,1]]

    if len(keys_not_matched_xml) > 0 and len(keys_not_matched_midi) > 0:
        match_quarter_grace, match_quarter_grace_note = match_note_arrays(score_xml_note_array[keys_not_matched_xml],
                                                score_midi_note_array[keys_not_matched_midi],
                          array_type='score', epsilon=1.0,
                          first_note_at_zero=True,
                          check_duration=False,
                          return_note_idxs=True)

        if len(match_quarter_grace_note) > 0:
            match_quarter_note = np.concatenate((match_quarter_note, match_quarter_grace_note), axis=0)

    # match the score midis
    match_sec, match_sec_note = match_note_arrays(array_score, perf_midi_note_array,
                      array_type='score', epsilon=0.05,
                      first_note_at_zero=True,
                      check_duration=False,
                      return_note_idxs=True)
    # match the performance midis
    match_perf, match_perf_note = match_note_arrays(ppart_midi_note_array, array_performance,
                      array_type='score', epsilon=0.05,
                      first_note_at_zero=True,
                      check_duration=False,
                      return_note_idxs=True)


    for note in alignment:
        name_in_score_corresp = note['score_id']
        # find this name in the corresp midi match
        pos_in_midi_match, = np.where(match_sec_note[:,0]==name_in_score_corresp)
        # extract the midi note name
        if len(pos_in_midi_match) > 0:
            name_in_score_midi = match_sec_note[pos_in_midi_match[0],1]
            # find this name in the xml midi match
            pos_in_score_match, = np.where(match_quarter_note[:,1]==name_in_score_midi)
            # insert the xml name in the alignment
            if len(pos_in_score_match) > 0:
                note['score_id'] = match_quarter_note[pos_in_score_match[0],0]

            elif len(pos_in_score_match) == 0:
                note['score_id'] = "*"
                note["label"] = "insertion"
        elif len(pos_in_midi_match) == 0:
            note['score_id'] = "*"
            note["label"] = "insertion"

        name_in_perf_corresp = note["performance_id"]
        pos_in_perf_match, = np.where(match_perf_note[:,1]==str(name_in_perf_corresp))
        if len(pos_in_perf_match) > 0:
            # insert the performance part id/name in the alignment
            note['performance_id'] = int(match_perf_note[pos_in_perf_match[0],0])

        elif len(pos_in_perf_match) == 0:
            note['performance_id'] = "*"
            note['label'] = "deletion"

    # CLEANUP ALIGNMENT; delete stuff that is in neither score nor performance
    alignment = [note for note in alignment if note['score_id'] != "*" or note['performance_id'] != "*"]

    return alignment, ppartp_midi, part_musicxml, (length_in_sec, length_in_quarter, average_tempo_sec_per_quarter)


if __name__ == '__main__':
    pass
    #test_corresp = "/Users/silvan/repos/chopin_cleaned-master/Beethoven/Piano_Sonatas/4-1/BENABD01_infer_corresp.txt"
    #test_smidi = "/Users/silvan/repos/chopin_cleaned-master/Beethoven/Piano_Sonatas/4-1/midi_cleaned.mid"
    #test_pmidi = "/Users/silvan/repos/chopin_cleaned-master/Beethoven/Piano_Sonatas/4-1/BENABD01.mid"
    #test_xml = "/Users/silvan/repos/chopin_cleaned-master/Beethoven/Piano_Sonatas/4-1/musicxml_cleaned.musicxml"
    #test_match_OUT = "/home/crow/repos/piano_performance_data/test/nakamura_ex/TEST.match"

    #a, b, c, l = alignment_from_corresp_pipeline(corresp_file = test_corresp,
    #                                                   performance_midi = test_pmidi,
    #                                                   score_midi = test_smidi,
    #                                                   musicxml = test_xml)

    #d = matchfile_from_alignment(a, b, c)
    #d.write(test_match_OUT)
