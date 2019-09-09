
import numpy as np
import mido

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


def load_midi(fn):
    """
    Load MIDI file and parse into Parts. (So far puts everything into
    one score part ??)

    Parameters
    ----------
    fn : string
        the file name

    Returns
    -------
    sp : ScorePart object
    """
    mid = MIDIFile(fn, unit='ticks', timing='absolute')
    part_id = 'Part 1'
    sp = score.Part(part_id)
    divs = score.Divisions(mid.ticks_per_beat)
    sp.timeline.add_starting_object(0, divs)

    # add notes
    for i, (onset, pitch, duration, velocity, channel) in enumerate(mid.notes):
        # TODO: use a pitch spelling approach
        # step, alter, octave = decode_pitch(pitch)
        step, alter, octave = decode_pitch_alternative(pitch)
        note_id = f's{i:04d}'
        note = score.Note(step, alter, octave, id=note_id)
        sp.timeline.add_starting_object(int(onset), note)
        sp.timeline.add_ending_object(int(onset + duration), note)

    # time signatures and measures
    time_sigs = mid.time_signatures.astype(np.int)
    # for convenience we add the end times for each time signature
    ts_end_times = np.r_[time_sigs[1:, 0], np.iinfo(np.int).max]
    time_sigs = np.column_stack((time_sigs, ts_end_times))

    measure_counter = 0
    for ts_start, num, den, ts_end in time_sigs:

        time_sig = score.TimeSignature(num, den)

        sp.timeline.add_starting_object(ts_start, time_sig)

        measure_duration = (num * mid.ticks_per_beat * 4) // den
        measure_start_limit = min(ts_end, sp.timeline.last_point.t)

        for m_start in range(ts_start, measure_start_limit, measure_duration):
            measure = score.Measure(number=measure_counter)
            m_end = min(m_start + measure_duration, ts_end)
            sp.timeline.add_starting_object(m_start, measure)
            sp.timeline.add_ending_object(m_end, measure)
            measure_counter += 1

        if np.isinf(ts_end):
            ts_end = m_end

        sp.timeline.add_ending_object(max(ts_start, min(ts_end, m_end)), time_sig)

    sp.timeline.add_ending_object(sp.timeline.last_point.t, divs)
    return sp

