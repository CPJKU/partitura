
import numpy as np
from mido import MidiFile, MidiTrack, Message

import partitura.score as score

import ipdb


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


def save_midi(fn, parts, ppq=DEFAULT_PPQ):
    """
    Write Parts to a MIDI file

     A type 0 file contains the entire performance, merged onto a single track,
     while type 1 files may contain any number of tracks that are performed
     in synchrony [https://en.wikipedia.org/wiki/MIDI#MIDI_files].


    NOTE: depending on how part looks like, we need to handle tracks,
    MIDI channels, etc.

    Parameters
    ----------
    part : score.Part() object

    """

    mf = MidiFile()  # create new file

    for part in parts:
        track = MidiTrack()
        mf.tracks.append(track)

        # set instrument/sound using MIDI program change
        track.append(Message('program_change', program=x, time=0))

        for note in part.notes:  # iterate over the part's notes
            track.append(Message('note_on', note=xx, velocity=yy, time=zz))
            track.append(Message('note_off', note=xx, velocity=yy, time=zz))


