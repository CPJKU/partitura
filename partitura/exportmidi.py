#!/usr/bin/env python

import logging
import numpy as np
from operator import itemgetter

from collections import defaultdict, OrderedDict
from mido import MidiFile, MidiTrack, Message

from partitura.score import iter_parts
from partitura.utils import partition

__all__ = ['save_midi']

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
                           for part in iter_parts(parts)])
    ppq = np.lcm.reduce(ppqs)
    return ppq


def save_midi(parts, out, part_voice_assign_mode=0, velocity=64):
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

    Returns
    -------
    
    """

    # TODO: write track names, time sigs, key sigs, tempos
    ppq = get_ppq(parts)

    onoffs = []
    onoff_keys = []
    onoff_key_set = OrderedDict()

    for i, part in enumerate(iter_parts(parts)):

        pg = get_partgroup(part)

        notes = part.notes_tied
        qm = part.quarter_map
        q_offset = qm(part.first_point.t)

        for note in notes:

            on = int((qm(note.start.t) - q_offset) * ppq)
            off = int((qm(note.end_tied.t) - q_offset) * ppq)

            onoffs.append(['note_on', on, note.midi_pitch])
            onoffs.append(['note_off', off, note.midi_pitch])

            # note_key is a tuple (part_group, part, voice)
            note_key = (pg, part, note.voice)
            # we add the note_key twice: once for note_on, once for note_off
            onoff_keys.extend([note_key, note_key])
            onoff_key_set[note_key] = True

    tr_ch_map = map_to_track_channel(list(onoff_key_set.keys()), part_voice_assign_mode)
    
    n_tracks = max(tr for tr, _ in tr_ch_map.values()) + 1
    tracks = [MidiTrack() for _ in range(n_tracks)]
    nows = [0]*n_tracks
    msg_order = np.argsort(np.fromiter((t for _, t, _ in onoffs),dtype=np.int))

    for i in msg_order:
        msg_key = onoff_keys[i]
        msg_type, t, pitch = onoffs[i]
        tr, ch = tr_ch_map[msg_key]
        delta = t - nows[tr]
        tracks[tr].append(Message(msg_type,
                                  note=pitch,
                                  channel=ch,
                                  velocity=velocity,
                                  time=delta))
        nows[tr] = t

    midi_type = 0 if n_tracks == 1 else 1
    
    mf = MidiFile(type=midi_type, ticks_per_beat=ppq)

    for track in tracks:
        mf.tracks.append(track)

    if out:
        if hasattr(out, 'write'):
            mf.save(file=out)
        else:
            mf.save(out)
