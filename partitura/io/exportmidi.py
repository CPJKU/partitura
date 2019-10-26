#!/usr/bin/env python

import logging
import numpy as np
from operator import itemgetter

from collections import defaultdict, OrderedDict
from mido import MidiFile, MidiTrack, Message, MetaMessage

import partitura.score as score
from partitura.utils import partition, MIDI_CONTROL_TYPES

__all__ = ['save_score_midi', 'save_performance_midi']

LOGGER = logging.getLogger(__name__)


def get_partgroup(part):
    parent = part
    while parent.parent:
        parent = parent.parent
    return parent


def map_to_track_channel(note_keys, mode):
    ch_helper = {}
    tr_helper = {}
    track = {}
    channel = {}
    for (pg, p, v) in note_keys:
        if mode == 0:
            trk = tr_helper.setdefault(p, len(tr_helper))
            ch1 = ch_helper.setdefault(p, {})
            ch2 = ch1.setdefault(v, len(ch1)+1)
            track[(pg, p, v)] = trk
            channel[(pg, p, v)] = ch2
        elif mode == 1:
            trk = tr_helper.setdefault(pg, len(tr_helper))
            ch1 = ch_helper.setdefault(pg, {})
            ch2 = ch1.setdefault(p, len(ch1)+1)
            track[(pg, p, v)] = trk
            channel[(pg, p, v)] = ch2
        elif mode == 2:
            track[(pg, p, v)] = 0
            ch = ch_helper.setdefault(p, len(ch_helper)+1)
            channel[(pg, p, v)] = ch
        elif mode == 3:
            trk = tr_helper.setdefault(p, len(tr_helper))
            track[(pg, p, v)] = trk
            channel[(pg, p, v)] = 1
        elif mode == 4:
            track[(pg, p, v)] = 0
            channel[(pg, p, v)] = 1
        elif mode == 5:
            trk = tr_helper.setdefault((p, v), len(tr_helper))
            track[(pg, p, v)] = trk
            channel[(pg, p, v)] = 1
        else:
            raise Exception('unsupported part/voice assign mode {}'
                            .format(mode))

    result = dict((k, (track.get(k, 0), channel.get(k, 1)))
                  for k in note_keys)
    # for (pg, p, voice), v in result.items():
    #     pgn = pg.group_name if hasattr(pg, 'group_name') else pg.id
    #     print(pgn, p.id, voice)
    #     print(v)
    #     print()
    return result
    

def get_ppq(parts):
    ppqs = np.concatenate([part.quarter_durations()[:, 1]
                           for part in score.iter_parts(parts)])
    ppq = np.lcm.reduce(ppqs)
    return ppq

def save_performance_midi(performed_part, out, mpq=500000, ppq=480, default_velocity=64):
    """Save a :class:`~partitura.performance.PerformedPart` instance as a
    MIDI file.

    Parameters
    ----------
    performed_part : :class:`~partitura.performance.PerformedPart`
        The performed part to save
    out : str or file-like object
        Either a filename or a file-like object to write the MIDI data
        to.
    mpq : int, optional
        Microseconds per quarter note. This is known in MIDI parlance
        as the "tempo" value. Defaults to 500000 (i.e. 120 BPM).
    ppq : int, optional
        Parts per quarter, also known as ticks per beat. Defaults to
        480.
    default_velocity : int, optional
        A default velocity value (between 0 and 127) to be used for
        notes without a specified velocity. Defaults to 64.

    """
    track_events = defaultdict(lambda: defaultdict(list))

    ct_to_int = dict((v, k) for k, v in MIDI_CONTROL_TYPES.items())
    for c in performed_part.controls:
        track = c.get('track', 0)
        ch = c.get('channel', 1)
        t = int(np.round(10**6*ppq*c['time']/mpq))
        track_events[track][t].append(
            Message('control_change', control=ct_to_int[c['type']], value=c['value'], channel=ch))

    for n in performed_part.notes:
        track = n.get('track', 0)
        ch = n.get('channel', 1)
        t_on = int(np.round(10**6*ppq*n['note_on']/mpq))
        t_off = int(np.round(10**6*ppq*n['note_off']/mpq))
        vel = n.get('velocity', default_velocity)
        track_events[track][t_on].append(
            Message('note_on', note=n['midi_pitch'], velocity=vel, channel=ch))
        track_events[track][t_off].append(
            Message('note_off', note=n['midi_pitch'], velocity=0, channel=ch))


    midi_type = 0 if len(track_events) == 1 else 1
    
    mf = MidiFile(type=midi_type, ticks_per_beat=ppq)

    for j, i in enumerate(sorted(track_events.keys())):
        track = MidiTrack()
        mf.tracks.append(track)
        if j == 0:
            track.append(MetaMessage('set_tempo', tempo=mpq, time=0))
        t = 0
        for t_msg in sorted(track_events[i].keys()):
            t_delta = t_msg - t
            for msg in track_events[i][t_msg]:
                track.append(msg.copy(time=t_delta))
                t_delta = 0
            t = t_msg
    if out:
        if hasattr(out, 'write'):
            mf.save(file=out)
        else:
            mf.save(out)


def save_score_midi(parts, out, part_voice_assign_mode=0, velocity=64):
    """Write data from Part objects to a MIDI file

    Parameters
    ----------
    parts : Part, PartGroup or list of these
        The musical score to be saved.
    out : str or file-like object
        Either a filename or a file-like object to write the MIDI data
        to.
    part_voice_assign_mode : {0, 1, 2, 3, 4, 5}, optional
        This keyword controls how part and voice information is
        associated to track and channel information in the MIDI file.
        The semantics of the modes is as follows:

        0
            Write one track for each Part, with channels assigned by
            voices
        1
            Write one track for each PartGroup, with channels assigned by
            Parts (voice info is lost) (There can be multiple levels of
            partgroups, I suggest using the highest level of
            partgroup/part) [note: this will e.g. lead to all strings into
            the same track] Each part not in a PartGroup will be assigned
            its own track
        2
            Write a single track with channels assigned by Part (voice
            info is lost)
        3
            Write one track per Part, and a single channel for all voices
            (voice info is lost)
        4
            Write a single track with a single channel (Part and voice
            info is lost)
        5
            Return one track per <Part, voice> combination, each track
            having a single channel.

    velocity : int, optional
        Default velocity for all MIDI notes.

    """

    ppq = get_ppq(parts)

    events = defaultdict(lambda: defaultdict(list))
    meta_events = defaultdict(lambda: defaultdict(list))

    event_keys = OrderedDict()
    tempos = {}

    for i, part in enumerate(score.iter_parts(parts)):
        
        pg = get_partgroup(part)

        notes = part.notes_tied
        qm = part.quarter_map
        q_offset = qm(part.first_point.t)

        def to_ppq(t):
            # convert div times to new ppq
            return int(ppq*qm(t))
        
        for tp in part.iter_all(score.Tempo):
            tempos[to_ppq(tp.start.t)] = MetaMessage('set_tempo', tempo=tp.microseconds_per_quarter)

        for ts in part.iter_all(score.TimeSignature):
            meta_events[part][to_ppq(ts.start.t)].append(MetaMessage('time_signature',
                                                                     numerator=ts.beats,
                                                                     denominator=ts.beat_type))

        for ks in part.iter_all(score.KeySignature):
            meta_events[part][to_ppq(ks.start.t)].append(MetaMessage('key_signature', key=ks.name))

        for note in notes:
            
            # key is a tuple (part_group, part, voice) that will be converted into a (track, channel) pair.
            key = (pg, part, note.voice)
            events[key][to_ppq(note.start.t)].append(Message('note_on', note=note.midi_pitch))
            events[key][to_ppq(note.end_tied.t)].append(Message('note_off', note=note.midi_pitch))

            event_keys[key] = True

    tr_ch_map = map_to_track_channel(list(event_keys.keys()),
                                     part_voice_assign_mode)


    # replace original event keys (partgroup, part, voice) by (track, ch) keys:
    for key in list(events.keys()):
        evs_by_time = events[key]
        del events[key]
        tr, ch = tr_ch_map[key]
        for t, evs in evs_by_time.items():
            events[tr][t].extend((ev.copy(channel=ch) for ev in evs))

    # figure out in which tracks to replicate the time/key signatures of each part
    part_track_map = partition(lambda x: x[0][1], tr_ch_map.items())
    for part, rest in part_track_map.items():
        part_track_map[part] = set(x[1][0] for x in rest)

    # add the time/key sigs to their corresponding tracks
    for part, m_events in meta_events.items():
        tracks = part_track_map[part]
        for tr in tracks:
            for t, me in m_events.items():
                events[tr][t] = me + events[tr][t]

    n_tracks = max(tr for tr, _ in tr_ch_map.values()) + 1
    tracks = [MidiTrack() for _ in range(n_tracks)]

    # tempo events are handled differently from key/time sigs because the have a
    # global effect. Instead of adding to each relevant track, like the key/time
    # sig events, we add them only to the first track
    track0_events = events[0]
    for t, tp in tempos.items():
        events[0][t].insert(0, tp)

    for tr, events_by_time in events.items():
        t_prev = 0
        for t in sorted(events_by_time.keys()):
            evs = events_by_time[t]
            delta = t - t_prev
            for ev in evs:
                tracks[tr].append(ev.copy(time=delta))
                delta = 0
            t_prev = t

    midi_type = 0 if n_tracks == 1 else 1
    
    mf = MidiFile(type=midi_type, ticks_per_beat=ppq)

    for track in tracks:
        mf.tracks.append(track)

    if out:
        if hasattr(out, 'write'):
            mf.save(file=out)
        else:
            mf.save(out)
