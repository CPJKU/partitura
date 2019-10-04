#!/usr/bin/env python

import logging
import numpy as np

from collections import defaultdict
from mido import MidiFile, MidiTrack, Message

from partitura.score import Part, PartGroup

import ipdb

__all__ = ['save_midi']
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


DEFAULT_FORMAT = 0  # MIDI file format
DEFAULT_PPQ = 480  # PPQ = pulses per quarter note
DEFAULT_TIME_SIGNATURE = (4, 4)


def get_parts_from_parts_partgroups(parts_partgroups, parts=[]):
    """
    From list of Part and/or PartGroup objects, get the Parts

    Parameters
    ----------
    parts_partgroups : list
        list of Part or PartGroup objects, the latter may containg
        further Part objects (CHECK this)

    parts : list, optional

    Returns
    -------
    parts : list of tuples (Part, int or None)
        (Part object, number of PartGroup or None)
        These are all parts present in `parts_partgroups`, extracted from
        their PartGroups and put into a list.
    """
    # pg = None
    for elem in parts_partgroups:
        if isinstance(elem, Part):
            # get last PartGroup that contains current Part
            pg_number = get_root(elem)
            parts.append((elem, pg_number))
        else:
            parts = get_parts_from_parts_partgroups(elem.children, parts)

    return parts


def get_root(part_partgroup):
    """
    Get a part's root PartGroup's number, if there is one.

    Parameters
    ----------
    part_partgroup : Part or PartGroup


    Returns
    -------
    number : int
    """
    p = part_partgroup
    number = None
    while isinstance(p.parent, PartGroup):
        p = p.parent
        number = p.number  # number of root PG

    return number


def assign_parts_voices_tracks_channels(mode, prt_grp_part_voice_list,
                                        onoff_list, prt_grp_part_voice_set):
    """
    Assign given parts and their voices' notes to MIDI tracks and channels.

    Parameters
    ----------
    mode : int

    prt_grp_part_voice_list : list of tuples

    onoff_list : list


    Returns
    -------
    assigned_notes : dictionary of lists
        each list has [note_on/note_off, time-stamp, midi_pitch, midi_channel],
        the dictionary keys are the numbers of the respective MIDI tracks
        the notes are assigned to.
    """
    chn_nr_start = 1    # check whether starts at zero or one
    track_nr_start = 1  # should start at 1

    assigned_notes = defaultdict(list)  # keys are track numbers

    if mode == 0:
        for kk, elem in enumerate(prt_grp_part_voice_list):
            assigned_notes[elem[1]].append([*onoff_list[kk], elem[2]])
    if mode == 1:
        pg_part_combis = set()
        only_part_combis = set()
        for elem in prt_grp_part_voice_set:
            if elem[0] is not None:
                pg_part_combis.add(elem[:2])
            elif elem[0] is None:
                only_part_combis.add(elem[:2])

        opc_sorted = list(only_part_combis)
        opc_sorted.sort(key=lambda x: x[1])

        ppc_sorted = list(pg_part_combis)
        ppc_sorted.sort(key=lambda x: (x[0], x[1]))

        track_nr_lookup = dict()
        for kk, elem in enumerate([*opc_sorted, *ppc_sorted]):
            track_nr_lookup[elem] = kk

        for kk, elem in enumerate(prt_grp_part_voice_list):
            assigned_notes[track_nr_lookup[tuple(elem[:2])]].append([*onoff_list[kk], elem[1]])

    if mode == 2:
        for kk, elem in enumerate(prt_grp_part_voice_list):
            # only one single track, number starting at 1
            # midi channels assigned by part number (they start at 1).
            assigned_notes[1].append([*onoff_list[kk], elem[1]])
    if mode == 3:
        for kk, elem in enumerate(prt_grp_part_voice_list):
            # each Part gets one track, all voices in one Part
            # put on same channel.
            assigned_notes[elem[1]].append([*onoff_list[kk], chn_nr_start])
    if mode == 4:
        for kk, elem in enumerate(prt_grp_part_voice_list):
            # only one single track, number starting at 1
            # assign all notes to the same channel.
            assigned_notes[1].append([*onoff_list[kk], chn_nr_start])
    if mode == 5:
        # all possible part, voice combis, removing PartGroup info
        # not sure if necessary
        part_voice_combis = set()
        for elem in prt_grp_part_voice_set:
            part_voice_combis.add(elem[1:])

        # should preserve order of Parts, voices
        pvc_sorted = list(part_voice_combis)
        pvc_sorted.sort(key=lambda x: (x[0], x[1]))

        # use the Part, voice combis for getting track numbers
        # they start at zero here
        track_nr_lookup = dict()
        for kk, elem in enumerate(pvc_sorted):
            track_nr_lookup[elem] = kk

        # finally, assign notes to the MIDI tracks
        for kk, elem in enumerate(prt_grp_part_voice_list):
            # single channel per track
            # the channel number is obtained from lookup dict
            assigned_notes[track_nr_lookup[tuple(elem[1:])]].append([*onoff_list[kk], chn_nr_start])

    for key in assigned_notes.keys():
        # sort by time-stamps all messages within each track.
        # is it enough to sort the notes by their time-stamp? Or should
        # there be some logic to make sure that the note_off always come
        # before a new note on of e.g. same pitch?
        # NOTE: as the note_off messages were added after each note_on
        # message into the list, I guess the sorting should keep that order
        # intact anyway?
        assigned_notes[key].sort(key=lambda x: x[0][1])

    return assigned_notes


def add_notes_to_track(assigned_notes_current_track, track, velocity):
    """Helper function for adding notes to a MIDI track

    Parameters
    ----------
    assigned_notes_current_track : list

    track : Mido MIDI track object (CHECK)

    Returns
    -------
    no returns
    """
    notes_by_ppq = defaultdict(list)
    for elem in assigned_notes_current_track:
        notes_by_ppq[elem[1]].append(elem)

    ts_sorted = sorted(notes_by_ppq.keys())  # somehow necessary
    last_ts = 0
    for ts in ts_sorted:  # notes_by_ppq.keys():
        delta_t = ts - last_ts
        for msg, divs, pitch, chn in notes_by_ppq[ts]:
            # print(f"delta_t: {delta_t}")
            # print(f"msg: {msg}")
            track.append(Message(msg,
                                 channel=chn,
                                 note=pitch,
                                 velocity=velocity,
                                 time=delta_t))
            delta_t = 0
        last_ts = ts


def save_midi(fn, parts_partgroups, part_voice_assign_mode=0, file_type=1,
              default_vel=64, ppq=DEFAULT_PPQ):
    """Write data from Part objects to a MIDI file

    Parameters
    ----------
    fn : str
        can this also be a file like object? Check

    parts_partgroups : single or list of mulitple score.Part objects
        CHECK!

    part_voice_assign_mode : {0, 1, 2, 3, 4, 5}, optional
        This keyword controls how part and voice information is associated
        to track and channel information in the MIDI file. The semantics of
        the modes is as follows:

        0
            Write one track for each Part, with channels assigned by voices
        1
            Write one track for each PartGroup, with channels assigned by Parts
            (voice info is lost)
            (There can be multiple levels of partgroups, I suggest using
            the highest level of partgroup/part) [note: this will e.g. lead
            to all strings into the same track]
            Each part not in a PartGroup will be assigned its own track
        2
            Write a single track with channels assigned by Part
            (voice info is lost)
        3
            Write one track per Part, and a single channel for all voices
            (voice info is lost)
        4
            Write a single track with a single channel
            (Part and voice info is lost)
        5
            Return one track per <Part, voice> combination,
            each track having a single channel.

    file_type : int, optional
        type of MIDI file to be created.

    default_vel : int, optional
        default velocity for all MIDI notes.

    ppq : int, optional.
        parts per quarter (ppq) for the MIDI file, i.e. amount of ticks
        per quarter note.

    Returns
    -------
    no returns
    """
    try:
        len(parts_partgroups)
    except TypeError:
        parts_partgroups = [parts_partgroups]  # wrap into list, makes things easier

    # get list of only Part objects
    parts_only = get_parts_from_parts_partgroups(parts_partgroups)

    onoff_list = []
    prt_grp_part_voice_list = []
    prt_grp_part_voice = set()  # all occurring different combinations
    for kk, elem in enumerate(parts_only):
        part = elem[0]
        pg_number = elem[1]

        # current part's notes, we use `notes_tied`!
        notes = part.notes_tied
        qm = part.quarter_map    # quarter map of the current part

        for note in notes:
            # check ints
            onoff_list.append(['note_on', int(qm(note.start.t) * ppq), note.midi_pitch])
            onoff_list.append(['note_off', int(qm(note.end_tied.t) * ppq), note.midi_pitch])

            # enumerate parts starting from 1
            # note that we do this 2 times to have equally many as note_on,
            # note_off messages.
            prt_grp_part_voice_list.append([pg_number, kk + 1, note.voice])
            prt_grp_part_voice_list.append([pg_number, kk + 1, note.voice])

            prt_grp_part_voice.add((pg_number, kk + 1, note.voice))

    # get mappings from prt_grp_part_voice (using a dict for lookup),
    # then partition onoff_list according to (trk). All notes of one track
    # combined, sorted by time stamp (per track) written to track. The channel
    # is part of note message -> mix into info from onoff_list.
    # Then fill tracks

    part_voice_assign_mode = 1  # remove this after testing!
    assigned_notes = assign_parts_voices_tracks_channels(part_voice_assign_mode,
                                                         prt_grp_part_voice_list,
                                                         onoff_list,
                                                         prt_grp_part_voice)

    # create object, spefify some basic parameters here
    mf = MidiFile(type=file_type, ticks_per_beat=ppq)

    for key in sorted(assigned_notes.keys()):  # keys are MIDI track numbers
        # print(key)
        # create track and append to file object
        track = MidiTrack()
        mf.tracks.append(track)

        # add all notes assigned to current track to the track object
        add_notes_to_track(assigned_notes[key], track, default_vel)

    mf.save(fn)  # save the midi file
