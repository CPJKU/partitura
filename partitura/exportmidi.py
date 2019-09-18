#!/usr/bin/env python

import logging
import numpy as np
from mido import MidiFile, MidiTrack, Message

from partitura.score import Divisions

import ipdb

__all__ = ['save_midi']
LOGGER = logging.getLogger(__name__)


DEFAULT_FORMAT = 0  # MIDI file format
DEFAULT_PPQ = 480  # PPQ = pulses per quarter note
DEFAULT_TIME_SIGNATURE = (4, 4)


STEPS = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B']


def decode_pitch_alternative(note_number):
    """

    Will only assign alter of 0 or 1.
    """
    # C4 at 261 Hz has step C, alter 0, octave 4.

    note_number = int(note_number)

    step_index = note_number % 12
    step = STEPS[step_index]

    if (step_index % 2 == 0):
        if (step_index <= 4):
            alter = 0
        else:
            alter = 1
    elif (step_index % 2 != 0):
        if (step_index > 4):
            alter = 0
        else:
            alter = 1

    octave = int(np.floor(note_number / 12 - 1))

    return step, alter, octave


def decode_pitch(pitch):
    """
    Naive approach to pitch-spelling. This is a place-holder
    for a proper pitch spelling approach.

    Parameters
    ----------
    pitch: int
        MIDI note number

    Returns
    -------
    (step, alter, octave): (str, int, int)
        Triple of step (note name), alter (0 or 1), and octave
    """

    octave = int((pitch - 12) // 12)
    step, alter = [('a', 0),
                   ('a', 1),
                   ('b', 0),
                   ('c', 0),
                   ('c', 1),
                   ('d', 0),
                   ('d', 1),
                   ('e', 0),
                   ('f', 0),
                   ('f', 1),
                   ('g', 0),
                   ('g', 1),
                   ][int((pitch - 21) % 12)]

    return step, alter, octave


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
    lcm_all_divs = np.lcm.reduce(list(all_divs))
    assert np.issubdtype(lcm_all_divs, np.integer)

    divs_factors = {}
    for div_val in all_divs:
        # Note: is there anywhere in score, etc., a check whether the div
        # values are actually integers? Then the whole testing here
        # would be unnecessary
        assert np.issubdtype(div_val, np.integer)  # check if best way
        divs_factors[div_val] = lcm_all_divs // div_val

        print(divs_factors[div_val])

    for part in parts:  # iterate over the parts
        # get array of all div values in current part
        divs = np.array([(divs.start.t, divs.divs) for divs in part.list_all(Divisions)])

        if len(divs) == 1:  # most likely case?
            part_divs = divs[0][1]
            part_divs_factor = divs_factors[divs[0][1]]

            divs_ppq_fact = ppq // (part_divs * part_divs_factor)

        # ipdb.set_trace()

        # basically: get the PPQ and scale all div values accordingly
        # so that the divisons per quarter are equal to PPQ. The note onset
        # times are given in delta PPQ since the last event always?

        track = MidiTrack()
        mf.tracks.append(track)

        # set instrument/sound using MIDI program change
        track.append(Message('program_change', program=0, time=0))

        cursor_position = 0
        for note in part.notes_tied:  # iterate over the part's notes
            note_start = int(note.start.t * divs_ppq_fact - cursor_position)
            note_end = int((note.end.t - note.start.t) * divs_ppq_fact)

            cursor_position = int(note.end.t * divs_ppq_fact)

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


