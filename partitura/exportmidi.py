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


    parts : list, optional


    Returns
    -------
    parts : list of tuples (Part, int or None)
        (Part object, number of PartGroup or None)
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


def assign_parts_voices_tracks_channels(mode, prt_grp_part_voice_list, onoff_list):
    """
    Assign given parts and their voices' notes to MIDI tracks and channels.

    Parameters
    ----------
    mode : int


    prt_grp_part_voice_list : list of tuples


    Returns
    -------
    assigned_notes : dictionary of lists
        each list has [[note_on/note_off, time-stamp, midi_pitch], midi_channel]

    """

    chn_nr = 0  # check whether starts at zero or one

    assigned_notes = defaultdict(list)  # keys are track numbers

    if mode == 0:
        for kk, elem in enumerate(prt_grp_part_voice_list):
            assigned_notes[elem[1]].append([*onoff_list[kk], elem[2]])
    if mode == 1:
        pass
    if mode == 2:
        for kk, elem in enumerate(prt_grp_part_voice_list):
            # only one track, number starting at 1
            # midi channels assigned by part number (they start at 1)
            assigned_notes[1].append([*[onoff_list[kk]], elem[1]])
    if mode == 3:
        pass
    if mode == 4:
        pass
    if mode == 5:
        pass

    for key in assigned_notes.keys():
        # is it enough to sort the notes by their time-stamp? Or should
        # there be some logic to make sure that the note_off always come
        # before a new note on of e.g. same pitch?
        assigned_notes[key].sort(key=lambda x: x[0][1])

    return assigned_notes


def add_note_to_track(track, channel, midi_pitch, velocity, note_start, note_end):
    """Helper function for adding notes to a track

    Parameters
    ----------


    Returns
    -------

    """
    track.append(Message('note_on',
                         channel=channel,
                         note=midi_pitch,
                         velocity=velocity,
                         time=note_start))
    track.append(Message('note_off',
                         channel=channel,
                         note=midi_pitch,
                         velocity=velocity,
                         time=note_end))


def save_midi(fn, parts_partgroups, part_voice_assign_mode=0, file_type=1,
              default_vel=64, ppq=DEFAULT_PPQ):
    """Write data from Part objects to a MIDI file

     A type 0 file contains the entire performance, merged onto a single track,
     while type 1 files may contain any number of tracks that are performed
     in synchrony [https://en.wikipedia.org/wiki/MIDI#MIDI_files].


    NOTE: depending on how part looks like, we need to handle tracks,
    MIDI channels, etc.

    Parameters
    ----------
    fn : str
        can this also be a file like object? Check

    parts_partgroups : single or list of mulitple score.Part objects

    part_voice_assign_mode : {0, 1, 2, 3, 4, 5}, optional
        This keyword controls how part and voice information is associated
        to track and channel information in the MIDI file. The semantics of
        the modes is as follows:

        0
            Write each Part's notes to a dedicated track, with voices mapped
            to MIDI channels
        1
            Each PartGroup's Parts are written to a single track,
            with Parts mapped to MIDI channels (possibly present voices
            are discarded)
        2
            Write single Part to tracks, where the Part's voices are
            mapped to tracks.
        3
            Write each Part to a track, possibly present voices are ignored,
            one single MIDI channel
        4

            Return single Part without voices (channel and track info is
            ignored)
        5
            Write each Part to a single track
            Return one Part per <track, channel> combination, without voices

        # Maarten's suggestions:

        0
            Write one track for each Part, with channels assigned by voices
        1
            Write one track for each PartGroup, with channels assigned by Parts
            (There can be multiple levels of partgroups, I suggest using
            the highest level of partgroup/part)
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

    file_type : int

    default_vel : int
        default velocity for all MIDI notes

    ppq : int
        parts per quarter (ppq) for the MIDI file, i.e. amount of ticks
        per quarter note.

    Returns
    -------

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

            prt_grp_part_voice.add((pg_number, kk, note.voice))

    # get mappings from prt_grp_part_voice (using a dict for lookup),
    # then partition onoff_list according to (trk). All notes of one track
    # combined, sorted by time stamp (per track) written to track. The channel
    # is part of note message -> mix into info from onoff_list.
    # Then fill tracks

    part_voice_assign_mode = 0  # remove this after testing!
    assigned_notes = assign_parts_voices_tracks_channels(part_voice_assign_mode,
                                                         prt_grp_part_voice_list,
                                                         onoff_list)

    # create object, spefify some basic parameters here
    mf = MidiFile(type=file_type, ticks_per_beat=ppq)

    for key in assigned_notes.keys():  # keys are MIDI tracks
        print(key)
        # create track and append to file object
        track = MidiTrack()
        mf.tracks.append(track)

        for elem in assigned_notes[key]:
            print(elem)


    ipdb.set_trace()


    # channel_cursor = 0

    # for part in parts:  # iterate over the parts
    #     # get array of all div values in current part
    #     divs = np.array([(divs.start.t, divs.divs) for divs in part.list_all(Divisions)])

    #     if len(divs) == 1:  # most likely case?
    #         part_divs = divs[0][1]
    #         part_divs_factor = divs_factors[divs[0][1]]

    #         LOGGER.debug(f"lcm: {lcm_all_divs}")
    #         LOGGER.debug(f"part_divs: {part_divs}")
    #         LOGGER.debug(f"part_divs_factor: {part_divs_factor}")

    #         assert ppq >= part_divs
    #         divs_ppq_fact = ppq // part_divs

    #         LOGGER.debug(f"divs_ppq_fact {divs_ppq_fact}")
    #     else:
    #         raise NotImplementedError()

    #     # basically: get the PPQ and scale all div values accordingly
    #     # so that the divisons per quarter are equal to PPQ. The note onset
    #     # times are given in delta PPQ since the last event always?

    #     track = MidiTrack()
    #     mf.tracks.append(track)

    #     # set instrument/sound using MIDI program change
    #     track.append(Message('program_change', program=0, time=0))

    #     notes_assigned = defaultdict(list)
    #     for note in part.notes_tied:  # this should incorporate the PPQ?
    #         notes_assigned[note.start.t].append(note)

    #     cursor_position = 0  # what if we have pickup measure? Should also work

    #     LOGGER.debug("\n")
    #     # note that the timeline should be manually unfolded first, if
    #     # necessary! Should this be done here, or where?

    #     # for note in part.notes_tied:  # iterate over the part's notes
    #     for key in notes_assigned:  # keys are timepoints
    #         LOGGER.debug(f"key: {key}")
    #         # for note in notes_assigned[key]:

    #         # take first note of possibly simultaneous notes
    #         note = notes_assigned[key][0]
    #         # common note start for all notes of same key
    #         note_start = int(note.start.t * divs_ppq_fact - cursor_position)

    #         longest_note_end = 0
    #         for note in notes_assigned[key]:
    #             LOGGER.debug(f"note: {note}")

    #             if note.voice is not None:
    #                 # mxml: 1, 2, ...; MIDI: 0, 1, ...;
    #                 channel_cursor = note.voice - 1
    #             LOGGER.debug(f"channel: {channel_cursor}")
    #             LOGGER.debug(f"voice: {note.voice}")

    #             # check the absoulute start and end times of notes, given in PPQ
    #             LOGGER.debug(f"note start TL: {note.start.t * divs_ppq_fact}")
    #             LOGGER.debug(f"note end TL: {note.end_tied.t * divs_ppq_fact}")

    #             LOGGER.debug(f"cursor at: {cursor_position}")

    #             # note_start = int(note.start.t * divs_ppq_fact - cursor_position)
    #             # delta to note start
    #             note_end = int((note.end_tied.t - note.start.t) * divs_ppq_fact - 1)

    #             if note_end > longest_note_end:  # longest duration is also latest endpoint
    #             # if int(note.end_tied.t * divs_ppq_fact) > longest_note_end:
    #                 longest_note_end = int(note.end_tied.t * divs_ppq_fact)

    #             LOGGER.debug(f"note start rel to cursor: {note_start}")
    #             LOGGER.debug(f"note end: {note_end}")

    #             # cursor_position = int(note.end_tied.t * divs_ppq_fact - 1)

    #             # add the note to the current track
    #             # add_note_to_track(track, channel_cursor, note.midi_pitch,
    #             #                   default_vel, note_start, note_end)

    #             # track.append(Message('note_on',
    #             #                      channel=channel_cursor,
    #             #                      note=note.midi_pitch,
    #             #                      velocity=default_vel,
    #             #                      time=note_start))
    #             # track.append(Message('note_off',
    #             #                      channel=channel_cursor,
    #             #                      note=note.midi_pitch,
    #             #                      velocity=default_vel,
    #             #                      time=note_end))

    #         # update cursor to duration of longest note
    #         cursor_position = int(longest_note_end - 1)
    #         LOGGER.debug(f"cursor updated to: {cursor_position}")


    ipdb.set_trace()
    mf.save(fn)  # save the midi file






