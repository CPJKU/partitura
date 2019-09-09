#!/usr/bin/env python
import argparse
import numpy as np
from collections import defaultdict
import warnings

import mido

import partitura.score as score
from partitura import save_musicxml
from partitura.utils import partition
from partitura.musicanalysis import estimate_spelling, estimate_key, estimate_voices

__all__ = ['load_midi']


def load_midi(fn, quantization_unit=None):
    # channels: parts, voices, staffs
    # tracks: parts, voices, staffs
    """
    Description
    
    Parameters
    ----------
    fn: type
        Description of `fn`
    quantization_unit: integer or None, optional
        If not None, quantize MIDI times to multiples of this unit.  . Defaults
        to None.
    
    Returns
    -------
    type
        Description of return value
    """
    
    mid = mido.MidiFile(fn)
    divs = mid.ticks_per_beat

    # as key for the dict use channel * 128 (max number of pitches) + pitch
    def note_hash(channel, pitch):
        """Generate a note hash."""
        return channel * 128 + pitch

    # these lists will contain information from dedicated tracks for meta
    # information (i.e. without notes)
    global_time_sigs = []
    global_key_sigs = []
    global_tempos = []

    # these dictionaries will contain meta information indexed by track (only
    # for tracks that contain notes)
    time_sigs_by_track = {}
    key_sigs_by_track = {}
    tempos_by_track = {}
    track_names_by_track = {}
    # notes are indexed by (track, channel) tuples
    notes_by_track_ch = {}

    for track_nr, track in enumerate(mid.tracks):
        time_sigs = []
        key_sigs = []
        # tempos = []
        notes = defaultdict(list)
        # dictionary for storing the last onset time and velocity for each
        # individual note (i.e. same pitch and channel)
        sounding_notes = {}
        # current time (will be updated by delta times in messages)
        t = 0

        for msg in track:

            # print(msg)

            t += msg.time

            if quantization_unit is not None:
                t = quantize(t, quantization_unit)
            
            if msg.is_meta:
                if msg.type == 'time_signature':
                    time_sigs.append((t, msg.numerator, msg.denominator))
                if msg.type == 'key_signature':
                    key_sigs.append((t, msg.key))
                if msg.type == 'set_tempo':
                    global_tempos.append((t, 60*10**6/msg.tempo))

            note_on = msg.type == 'note_on'
            note_off = msg.type == 'note_off'

            if not (note_on or note_off):
                continue

            # hash sounding note
            note = note_hash(msg.channel, msg.note)

            # start note if it's a 'note on' event with velocity > 0
            if note_on and msg.velocity > 0:

                # save the onset time and velocity
                sounding_notes[note] = (t, msg.velocity)

            # end note if it's a 'note off' event or 'note on' with velocity 0
            elif note_off or (note_on and msg.velocity == 0):

                if note not in sounding_notes:
                    warnings.warn('ignoring MIDI message %s' % msg)
                    continue

                # append the note to the list associated with the channel
                notes[msg.channel].append((sounding_notes[note][0], msg.note, t-sounding_notes[note][0]))
                                          # sounding_notes[note][1]])
                # remove hash from dict
                del sounding_notes[note]

        # if a track has no notes, we assume it may contain global time/key sigs
        # and tempo values
        if not notes:
            global_time_sigs.extend(time_sigs)
            global_key_sigs.extend(key_sigs)
            # global_tempos.extend(tempos)

        for ch, ch_notes in notes.items():
            # if there are any notes, store the notes along with key sig / time
            # sig / tempo information under the key (track_nr, ch_nr)
            if len(ch_notes) > 0:
                notes_by_track_ch[(track_nr, ch)] = ch_notes
                time_sigs_by_track[track_nr] = time_sigs
                key_sigs_by_track[track_nr] = key_sigs
                # tempos_by_track[track_nr] = tempos
                track_names_by_track[track_nr] = track.name

    part_keys = sorted(notes_by_track_ch.keys())

    # pitch spelling, voice estimation and key estimation are done on a
    # structured array (onset, pitch, duration) of all notes in the piece
    # jointly, so we concatenate all notes
    # note_list = sorted(note for notes in (notes_by_track_ch[key] for key in part_keys) for note in notes)
    note_list = [note for notes in (notes_by_track_ch[key] for key in part_keys) for note in notes]
    note_array = np.array(note_list, dtype=[('onset', np.int), ('pitch', np.int), ('duration', np.int)])

    # do pitch spelling
    step, alter, octave = estimate_spelling(note_array)
    # convert spelling to struct array, this should be inside estimate_spelling
    spelling = np.empty(len(step), dtype=[('step', 'U1'), ('alter', np.int), ('octave', np.int)])
    spelling['step'] = step
    spelling['alter'] = alter
    spelling['octave'] = octave

    # add the spellings to their corresponding parts
    offset = 0
    spellings_by_track_ch = {}
    for key in part_keys:
        N = len(notes_by_track_ch[key])
        spellings_by_track_ch[key] = spelling[offset:offset+N]
        offset += N

    part_keys_by_track = partition(lambda x: x[0], part_keys)

    parts = []
    for track, track_keys in part_keys_by_track.items():
        track_parts = []
        for track, ch in track_keys:
            track_parts.append(create_part(divs,
                                           notes_by_track_ch[(track, ch)],
                                           spellings_by_track_ch[(track, ch)],
                                           time_sigs_by_track[track] or global_time_sigs,
                                           key_sigs_by_track[track] or global_key_sigs,
                                           # tempos_by_track[track] or global_tempos,
                                           part_id='P{}'.format(len(parts)+1),
                                           part_name=(track_names_by_track[track]
                                                      or 'Track {}, Channel {}'.format(track, ch))))

        if len(track_parts) == 1:
            # if there is only one channel in a track, we don't embed the Part in a PartGroup
            parts.append(track_parts[0])
        elif len(part_keys_by_track) == 1:
            parts.extend(track_parts)
        else:
            part_group = score.PartGroup('brace', track_names_by_track[track])
            part_group.children = track_parts
            parts.append(part_group)

    # add tempos to first part
    part = next(score.iter_parts(parts))
    for t, qpm in global_tempos:
        part.add(t, score.Tempo(qpm, unit='q'))
        
    return parts


def create_part(ticks, notes, spellings, time_sigs, key_sigs, part_id=None, part_name=None):
    part = score.Part(part_id)
    part.add(0, score.Divisions(ticks))
    # for t, name in key_sigs:
    #     part.add(t, score.TimeSignature(num, den))
    for (onset, pitch, duration), (step, alter, octave) in zip(notes, spellings):
        note = score.Note(step, alter, octave)
                          # symbolic_duration=score.estimate_symbolic_duration(duration, ticks))
        part.add(onset, note, onset+duration)

    if not time_sigs:
        warnings.warn('No time signatures found, assuming 4/4')
        time_sigs = [(0, 4, 4)]

    time_sigs = np.array(time_sigs, dtype=np.int)

    # for convenience we add the end times for each time signature
    ts_end_times = np.r_[time_sigs[1:, 0], np.iinfo(np.int).max]
    time_sigs = np.column_stack((time_sigs, ts_end_times))
    
    measure_counter = 1
    for ts_start, num, den, ts_end in time_sigs:

        time_sig = score.TimeSignature(num, den)

        part.add(ts_start, time_sig)
        
        measure_duration = (num * ticks * 4) // den
        measure_start_limit = min(ts_end, part.timeline.last_point.t)

        for m_start in range(ts_start, measure_start_limit, measure_duration):
            measure = score.Measure(number=measure_counter)
            m_end = min(m_start+measure_duration, ts_end)
            part.add(m_start, measure, m_end)
            measure_counter += 1

        if np.isinf(ts_end):
            ts_end = m_end
        
    # tie notes spanning measure boundaries
    tie_notes(part)

    return part


def tie_notes(part):
    # split and tie notes at measure boundaries
    notes = part.list_all(score.Note)
    for note in notes:
        next_measure = next(iter(note.start.get_next_of_type(score.Measure)), None)
        note_end = note.end
        while next_measure and note.end > next_measure.start:
            part.timeline.remove_ending_object(note)
            part.timeline.add_ending_object(next_measure.start.t, note)
            tie_next = score.Note(note.step, note.alter, note.octave, voice=note.voice, staff=note.staff)
            part.add(next_measure.start.t, tie_next)
            part.timeline.add_ending_object(note_end.t, tie_next)
            note.tie_next = tie_next
            tie_next.tie_prev = note
            note = tie_next
            next_measure = next(iter(note.start.get_next_of_type(score.Measure)), None)


    # then split/tie any notes that do not have a fractional/dot duration
    divs_map = part.divisions_map
    notes = part.list_all(score.Note)

    for note in notes:

        if note.symbolic_duration is None:

            splits = score.find_tie_split(note.start.t, note.end.t, divs_map(note.start.t))

            if splits:

                split_note(part, note, splits)


def split_note(part, note, splits):
    # TODO: we shouldn't do this, but for now it's a good sanity check
    assert len(splits) > 0
    # TODO: we shouldn't do this, but for now it's a good sanity check
    assert note.symbolic_duration is None
    part.remove(note)

    cur_note = note
    start, end, sym_dur = splits.pop(0)
    part.add(start, cur_note, end)

    while splits:
        next_note = score.Note(note.step, note.alter, note.octave, voice=note.voice, staff=note.staff)
        cur_note.tie_next = next_note
        next_note.tie_prev = cur_note

        cur_note = next_note
        start, end, sym_dur = splits.pop(0)
        part.add(start, cur_note, end)


def quantize(v, unit):
    """
    Quantize value `v` to a multiple of `unit`. When `unit` is an integer, the
    return value will be integer as well, otherwise the function will return a
    float.
    
    Parameters
    ----------
    v: ndarray or number
        Number to be quantized
    unit: number
        The quantization unit

    Returns
    -------
    number
        The quantized number

    Examples
    --------

    >>> quantize(13.3, 4)
    12

    >>> quantize(3.3, .5)
    3.5
    """
    
    r = unit * np.round(v / unit)
    if isinstance(unit, int):
        return int(r)
    else:
        return r


if __name__ == '__main__':
    # print(find_tie_split(1, 14, 4))
    import doctest
    doctest.testmod()

