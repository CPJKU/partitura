#!/usr/bin/env python

import logging
import numpy as np
from mido import MidiFile, MidiTrack, Message

from partitura.score import Divisions

import ipdb

__all__ = ['save_midi']
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)



DEFAULT_FORMAT = 0  # MIDI file format
DEFAULT_PPQ = 480  # PPQ = pulses per quarter note
DEFAULT_TIME_SIGNATURE = (4, 4)


def save_midi(fn, parts, file_type=1, default_vel=64, ppq=DEFAULT_PPQ):
    """
    Write data from Part objects to a MIDI file

     A type 0 file contains the entire performance, merged onto a single track,
     while type 1 files may contain any number of tracks that are performed
     in synchrony [https://en.wikipedia.org/wiki/MIDI#MIDI_files].


    NOTE: depending on how part looks like, we need to handle tracks,
    MIDI channels, etc.

    Parameters
    ----------
    fn : str
        can this also be a file like object? Check

    parts : single or list of mulitple score.Part objects

    """
    mf = MidiFile(type=file_type, ticks_per_beat=ppq)  # create new file

    # get from all parts their Divisions values and find a value that works
    # for all parts.
    divs_list = []  # check which part has which divison setting
    for part in parts:  # iterate over the parts
        # get an array of the current part's divs
        divs = np.array([(divs.start.t, divs.divs) for divs in part.list_all(Divisions)])
        divs_list.append(divs[:, 1])  # keep only the div values

    all_divs = set(np.array(divs_list).flatten())
    # get the least common multiple of all involved div values used in the 
    lcm_all_divs = np.lcm.reduce(list(all_divs))
    assert np.issubdtype(lcm_all_divs, np.integer)

    divs_factors = {}
    for div_val in all_divs:
        # Note: is there anywhere in score, etc., a check whether the div
        # values are actually integers? Then the whole testing here
        # would be unnecessary
        assert np.issubdtype(div_val, np.integer)  # check if best way
        divs_factors[div_val] = lcm_all_divs // div_val

        # print(divs_factors[div_val])

    for part in parts:  # iterate over the parts
        # get array of all div values in current part
        divs = np.array([(divs.start.t, divs.divs) for divs in part.list_all(Divisions)])

        if len(divs) == 1:  # most likely case?
            part_divs = divs[0][1]
            part_divs_factor = divs_factors[divs[0][1]]

            print(f"lcm: {lcm_all_divs}")
            print(f"part_divs: {part_divs}")
            print(f"part_divs_factor: {part_divs_factor}")

            # divs_ppq_fact = ppq // (part_divs * part_divs_factor)
            assert ppq >= part_divs
            divs_ppq_fact = ppq // part_divs

            print(f"divs_ppq_fact {divs_ppq_fact}")
        else:
            raise NotImplementedError()


        # basically: get the PPQ and scale all div values accordingly
        # so that the divisons per quarter are equal to PPQ. The note onset
        # times are given in delta PPQ since the last event always?

        track = MidiTrack()
        mf.tracks.append(track)

        # set instrument/sound using MIDI program change
        track.append(Message('program_change', program=0, time=0))

        cursor_position = 0  # what if we have pickup measure? Should also work
        for note in part.notes_tied:  # iterate over the part's notes
            LOGGER.debug(f"cursor at: {cursor_position}")

            note_start = int(note.start.t * divs_ppq_fact - cursor_position)
            note_end = int((note.end.t - note.start.t) * divs_ppq_fact - 1)

            LOGGER.debug(f"note start: {note_start}")
            LOGGER.debug(f"note end: {note_end}")

            cursor_position = int(note.end.t * divs_ppq_fact - 1)

            LOGGER.debug(f"cursor updated to: {cursor_position}")

            # ipdb.set_trace()

            track.append(Message('note_on',
                                 note=note.midi_pitch,
                                 velocity=default_vel,
                                 time=note_start))
            track.append(Message('note_off',
                                 note=note.midi_pitch,
                                 velocity=default_vel,
                                 time=note_end))

    mf.save(fn)  # save the midi file


