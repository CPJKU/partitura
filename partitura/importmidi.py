#!/usr/bin/env python
import argparse
import numpy as np
from collections import defaultdict
import warnings
import logging
from scipy.interpolate import interp1d

import mido

import partitura.score as score
from partitura import save_musicxml
from partitura.utils import partition, estimate_symbolic_duration, key_name_to_fifths_mode, fifths_mode_to_key_name
import partitura.musicanalysis as analysis

__all__ = ['load_midi']
LOGGER = logging.getLogger(__name__)


def load_midi(fn, part_voice_assign_mode=0, ensure_list=False,
              quantization_unit=None, estimate_voice_info=True,
              estimate_key=False, assign_note_ids=True):

    """Load a musical score from a MIDI file. Pitch names are estimated
    using Meredith's PS13 algorithm [1]_.


    Parameters
    ----------
    fn : type
        Description of `fn`
    part_voice_assign_mode : {0, 1, 2, 3, 4, 5}, optional
        This keyword controls how part and voice information is associated
        to track and channel information in the MIDI file. The semantics of
        the modes is as follows:

        0
            Return one Part per track, with voices assigned by channel
        1
            Return one PartGroup per track, with Parts assigned by channel (no
            voices)
        2
            Return single Part with voices assigned by track (tracks are
            combined, channel info is ignored)
        3
            Return one Part per track, without voices (channel info is ignored)
        4
            Return single Part without voices (channel and track info is
            ignored)
        5
            Return one Part per <track, channel> combination, without voices

        Defaults to 0.
    ensure_list : bool, optional
        When True, return a list independent of how many part or partgroup
        elements were created from the MIDI file. By default, when the
        return value of `load_midi` produces a single
        :class:`partitura.score.Part` or :class:`partitura.score.PartGroup`
        element, the element itself is returned instead of a list
        containing the element. Defaults to False.
    quantization_unit : integer or None, optional
        Quantize MIDI times to multiples of this unit. If None, the
        quantization unit is chosen automatically as the smallest division
        of the parts per quarter (MIDI "ticks") that can be represented as
        a symbolic duration. Defaults to None.
    estimate_key : bool, optional
        When True use Krumhansl's 1990 key profiles [3]_ to determine the
        most likely global key, discarding any key information in the MIDI
        file.
    estimate_voice_info : bool, optional
        When True use Chew and Wu's voice separation algorithm [2]_ to
        estimate voice information. This option is ignored for
        part/voice assignment modes that infer voice information from
        the track/channel info (i.e. `part_voice_assign_mode` equals
        1, 3, 4, or 5). Defaults to True.

    Returns
    -------
    :class:`partitura.score.Part`, :class:`partitura.score.PartGroup`, or a list of these
        One or more part or partgroup objects

    References
    ----------

    .. [1] Dave Meredith, "The ps13 Pitch Spelling Algorithm", Journal of
           New Music Research 35(2), 2006.
    .. [2] Elaine Chew and Xiaodan Wu, "Separating Voices in Polyphonic
           Music: A Contig Mapping Approach". Computer Music Modeling and
           Retrieval (CMMR), pp. 1–20, 2005.
    .. [3] Carol L. Krumhansl, "Cognitive foundations of musical pitch",
           Oxford University Press, New York, 1990.

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
    relevant = {'time_signature',
                'key_signature',
                'set_tempo',
                'note_on',
                'note_off'}
    for track_nr, track in enumerate(mid.tracks):
        time_sigs = []
        key_sigs = []
        # tempos = []
        notes = defaultdict(list)
        # dictionary for storing the last onset time and velocity for each
        # individual note (i.e. same pitch and channel)
        sounding_notes = {}
        # current time (will be updated by delta times in messages)
        t_raw = 0

        for msg in track:

            t_raw = t_raw + msg.time

            if msg.type not in relevant:
                continue

            if quantization_unit:
                t = quantize(t_raw, quantization_unit)
            else:
                t = t_raw

            if msg.type == 'time_signature':
                time_sigs.append((t, msg.numerator, msg.denominator))
            if msg.type == 'key_signature':
                key_sigs.append((t, msg.key))
            if msg.type == 'set_tempo':
                global_tempos.append((t, 60*10**6/msg.tempo))
            else:
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
        if not notes:
            global_time_sigs.extend(time_sigs)
            global_key_sigs.extend(key_sigs)
        else:
            # if there are note, we store the info under the track number
            time_sigs_by_track[track_nr] = time_sigs
            key_sigs_by_track[track_nr] = key_sigs
            track_names_by_track[track_nr] = track.name

        for ch, ch_notes in notes.items():
            # if there are any notes, store the notes along with key sig / time
            # sig / tempo information under the key (track_nr, ch_nr)
            if len(ch_notes) > 0:
                notes_by_track_ch[(track_nr, ch)] = ch_notes

    tr_ch_keys = sorted(notes_by_track_ch.keys())
    group_part_voice_keys, part_names, group_names = assign_group_part_voice(part_voice_assign_mode,
                                                                             tr_ch_keys,
                                                                             track_names_by_track)
    # for key and time sigs:
    track_to_part_mapping = make_track_to_part_mapping(tr_ch_keys, group_part_voice_keys)

    # pairs of (part, voice) for each note
    part_voice_list = [[part, voice] for tr_ch, (_, part, voice)
                       in zip(tr_ch_keys, group_part_voice_keys)
                       for i in range(len(notes_by_track_ch[tr_ch]))]

    # pitch spelling, voice estimation and key estimation are done on a
    # structured array (onset, pitch, duration) of all notes in the piece
    # jointly, so we concatenate all notes
    # note_list = sorted(note for notes in (notes_by_track_ch[key] for key in tr_ch_keys) for note in notes)
    note_list = [note for notes in (notes_by_track_ch[key] for key in tr_ch_keys) for note in notes]
    note_array = np.array(note_list, dtype=[('onset', np.int), ('pitch', np.int), ('duration', np.int)])

    LOGGER.debug('pitch spelling')
    spelling_global = analysis.estimate_spelling(note_array)

    if estimate_voice_info:
        LOGGER.debug('voice estimation')
        estimated_voices = analysis.estimate_voices(note_array)
        # TODO: don't do +1 as soon as this is done in estimate_voices
        estimated_voices += 1
        for part_voice, voice_est in zip(part_voice_list, estimated_voices):
            if part_voice[1] is None:
                part_voice[1] = voice_est

    if estimate_key:
        LOGGER.debug('key estimation')
        _, mode, fifths = analysis.estimate_key(note_array)
        key_sigs_by_track = {}
        global_key_sigs = [(0, fifths_mode_to_key_name(fifths, mode))]

    if assign_note_ids:
        note_ids = ['n{}'.format(i) for i in range(len(note_array))]
    else:
        note_ids = [None for i in range(len(note_array))]

    time_sigs_by_part = defaultdict(set)
    for tr, ts_list in time_sigs_by_track.items():
        for ts in ts_list:
            for part in track_to_part_mapping[tr]:
                time_sigs_by_part[part].add(ts)
    for ts in global_time_sigs:
        for part in set(part for _, part, _ in group_part_voice_keys):
            time_sigs_by_part[part].add(ts)

    key_sigs_by_part = defaultdict(set)
    for tr, ks_list in key_sigs_by_track.items():
        for ks in ks_list:
            for part in track_to_part_mapping[tr]:
                key_sigs_by_part[part].add(ks)
    for ks in global_key_sigs:
        for part in set(part for _, part, _ in group_part_voice_keys):
            key_sigs_by_part[part].add(ks)

    # names_by_part = defaultdict(set)
    # for tr_ch, pg_p_v in zip(tr_ch_keys, group_part_voice_keys):
    #     print(tr_ch, pg_p_v)
    # for tr, name in track_names_by_track.items():
    #     print(tr, track_to_part_mapping, name)
    #     for part in track_to_part_mapping[tr]:
    #         names_by_part[part] = name

    notes_by_part = defaultdict(list)
    for (part, voice), note, spelling, note_id in zip(part_voice_list,
                                                      note_list,
                                                      spelling_global,
                                                      note_ids):
        notes_by_part[part].append((note, voice, spelling, note_id))

    partlist = []
    part_to_part_group = dict((p, pg) for pg, p, _ in group_part_voice_keys)
    part_groups = {} 
    for part_nr, note_info in notes_by_part.items():
        notes, voices, spellings, note_ids = zip(*note_info)
        part = create_part(divs, notes, spellings, voices, note_ids,
                           sorted(time_sigs_by_part[part_nr]),
                           sorted(key_sigs_by_part[part_nr]),
                           part_id='P{}'.format(part_nr+1),
                           part_name=part_names.get(part_nr, None))

        # print(part.pretty())
        # if this part has an associated part_group number we create a PartGroup
        # if necessary, and add the part to that. The newly created PartGroup is
        # then added to the partlist.
        pg_nr = part_to_part_group[part_nr]
        if pg_nr is None:
            partlist.append(part)
        else:
            if pg_nr not in part_groups:
                part_groups[pg_nr] = score.PartGroup(group_name=group_names.get(pg_nr, None))
                partlist.append(part_groups[pg_nr])
            part_groups[pg_nr].children.append(part)

    # add tempos to first part
    part = next(score.iter_parts(partlist))
    for t, qpm in global_tempos:
        part.add(score.Tempo(qpm, unit='q'), t)

    if not ensure_list and len(partlist) == 1:
        return partlist[0]
    else:
        return partlist


def make_track_to_part_mapping(tr_ch_keys, group_part_voice_keys):
    """Return a mapping from track numbers to one or more parts. This mapping tells
    us where to put meta event info like time and key sigs.
    """
    track_to_part_keys = defaultdict(set)
    for (tr, _), (_, part, _) in zip(tr_ch_keys, group_part_voice_keys):
        track_to_part_keys[tr].add(part)
    return track_to_part_keys

# def timings_ok(durations, divs, threshold=.1):
#     n_without_dur = sum(1 for dur in durations if not estimate_symbolic_duration(dur, divs))
#     prop_without_dur = n_without_dur/max(1, len(durations))
#     if prop_without_dur > threshold:
#         LOGGER.warning('{:.1f}% of the notes ({}/{}) have irregular durations. Maybe you want to load this file as a performance rather than a score. If you do wish to interpret the MIDI as a score use the option --force-duration-analysis, but beware that analysis may be very slow and still fail. Another option is to quantize note onset and offset times by setting the `quantization_unit` keyword argument of `load_midi`) to an appropriate value'.format(100*prop_without_dur, n_without_dur, len(durations)))
#     return prop_without_dur < threshold


def assign_group_part_voice(mode, track_ch_combis, track_names):
    """
    0: return one Part per track, with voices assigned by channel
    1. return one PartGroup per track, with Parts assigned by channel (no voices)
    2. return single Part with voices assigned by track (tracks are combined, channel info is ignored)
    3. return one Part per track, without voices (channel info is ignored)
    4. return single Part without voices (channel and track info is ignored)
    5. return one Part per <track, channel> combination, without voices
    """
    part_group = {}
    part = {}
    voice = {}
    part_helper = {}
    voice_helper = {}
    part_group_helper = {}

    part_names = {}
    group_names = {}
    for tr, ch in track_ch_combis:
        if mode == 0:
            prt = part_helper.setdefault(tr, len(part_helper))
            vc1 = voice_helper.setdefault(tr, {})
            vc2 = vc1.setdefault(ch, len(vc1) + 1)
            part_names[prt] = '{}'.format(track_names.get(tr, 'Track {}'.format(tr+1)))
            part[(tr, ch)] = prt
            voice[(tr, ch)] = vc2
        elif mode == 1:
            pg = part_group_helper.setdefault(tr, len(part_group_helper))
            prt = part_helper.setdefault(ch, len(part_helper))
            part_group.setdefault((tr, ch), pg)
            # group_names[pg] = '{}'.format(track_names.get(tr, 'Track {}'.format(tr+1)), ch)
            group_names[pg] = track_names.get(tr, 'Track {}'.format(tr+1))
            part_names[prt] = 'ch={}'.format(ch)
            part[(tr, ch)] = prt
        elif mode == 2:
            vc = voice_helper.setdefault(tr, len(voice_helper) + 1)
            part.setdefault((tr, ch), 0)
            voice[(tr, ch)] = vc
        elif mode == 3:
            prt = part_helper.setdefault(tr, len(part_helper))
            part_names[prt] = '{}'.format(track_names.get(tr, 'Track {}'.format(tr+1)))
            part[(tr, ch)] = prt
        elif mode == 4:
            part.setdefault((tr, ch), 0)
        elif mode == 5:
            part_names[(tr, ch)] = '{} ch={}'.format(track_names.get(tr, 'Track {}'.format(tr+1)), ch)
            part.setdefault((tr, ch), len(part))

    return [(part_group.get(tr_ch), part.get(tr_ch), voice.get(tr_ch))
            for tr_ch in track_ch_combis], part_names, group_names


def estimate_clef(pitches):
    # avg_pitch = np.mean(pitches)
    center = np.median(pitches)
    # number, sign, line, octave_change):
    clefs = [score.Clef(1, 'F', 4, 0), score.Clef(1, 'G', 2, 0)]
    f = interp1d([0, 49, 70, 127], [0, 0, 1, 1], kind='nearest')
    return clefs[int(f(center))]


def create_part(ticks, notes, spellings, voices, note_ids, time_sigs, key_sigs, part_id=None, part_name=None):
    LOGGER.debug('create_part')

    part = score.Part(part_id, part_name=part_name)
    part.set_quarter_duration(0, ticks)

    clef = estimate_clef([pitch for _, pitch, _ in notes])
    part.add(clef, 0)
    for t, name in key_sigs:
        fifths, mode = key_name_to_fifths_mode(name)
        part.add(score.KeySignature(fifths, mode), t)

    LOGGER.debug('add notes')

    for (onset, pitch, duration), (step, alter, octave), voice, note_id in zip(notes, spellings, voices, note_ids):
        if duration > 0:
            note = score.Note(step, octave, alter, voice=int(voice or 0), id=note_id,
                              symbolic_duration=estimate_symbolic_duration(duration, ticks))
        else:
            note = score.GraceNote('appoggiatura', step, octave, alter, voice=int(voice or 0), id=note_id,
                                   symbolic_duration=dict(type='quarter'))

        part.add(note, onset, onset+duration)

    if not time_sigs:
        warnings.warn('No time signatures found, assuming 4/4')
        time_sigs = [(0, 4, 4)]

    time_sigs = np.array(time_sigs, dtype=np.int)

    # for convenience we add the end times for each time signature
    ts_end_times = np.r_[time_sigs[1:, 0], np.iinfo(np.int).max]
    time_sigs = np.column_stack((time_sigs, ts_end_times))

    LOGGER.debug('add time sigs and measures')

    for ts_start, num, den, ts_end in time_sigs:
        time_sig = score.TimeSignature(num.item(), den.item())
        part.add(time_sig, ts_start.item())

    score.add_measures(part)

    # this is the old way to add measures. Since part comes from MIDI we
    # only have a single global divs value, which makes add it easier to compute
    # measure durations:
    
    # measure_counter = 1
    # # we call item() on numpy numbers to get the value in the equivalent python type
    # for ts_start, num, den, ts_end in time_sigs:
    #     time_sig = score.TimeSignature(num.item(), den.item())
    #     part.add(time_sig, ts_start.item())
    #     measure_duration = (num.item() * ticks * 4) // den.item()
    #     measure_start_limit = min(ts_end.item(), part.last_point.t)
    #     for m_start in range(ts_start, measure_start_limit, measure_duration):
    #         measure = score.Measure(number=measure_counter)
    #         m_end = min(m_start+measure_duration, ts_end)
    #         part.add(measure, m_start, m_end)
    #         measure_counter += 1
    #     if np.isinf(ts_end):
    #         ts_end = m_end

    LOGGER.debug('tie notes')
    # tie notes where necessary (across measure boundaries, and within measures
    # notes with compound duration)
    score.tie_notes(part)

    LOGGER.debug('find tuplets')
    # apply simplistic tuplet finding heuristic
    score.find_tuplets(part)

    LOGGER.debug('done create_part')
    return part


def quantize(v, unit):
    """Quantize value `v` to a multiple of `unit`. When `unit` is an integer,
    the return value will be integer as well, otherwise the function will
    return a float.

    Parameters
    ----------
    v : ndarray or number
        Number to be quantized
    unit : number
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
    import doctest
    doctest.testmod()
