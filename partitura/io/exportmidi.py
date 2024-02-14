#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for exporting MIDI files
"""
import numpy as np

from collections import defaultdict, OrderedDict
from typing import Optional, Iterable

from mido import MidiFile, MidiTrack, Message, MetaMessage, merge_tracks

import partitura.score as score
from partitura.score import Score, Part, PartGroup, ScoreLike
from partitura.performance import Performance, PerformedPart, PerformanceLike
from partitura.utils import partition, fifths_mode_to_key_name

from partitura.utils.misc import deprecated_alias, PathLike

__all__ = ["save_score_midi", "save_performance_midi"]


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
    for pg, p, v in note_keys:
        if mode == 0:
            trk = tr_helper.setdefault(p, len(tr_helper))
            ch1 = ch_helper.setdefault(p, {})
            ch2 = ch1.setdefault(v, len(ch1) + 1)
            track[(pg, p, v)] = trk
            channel[(pg, p, v)] = ch2
        elif mode == 1:
            trk = tr_helper.setdefault(pg, len(tr_helper))
            ch1 = ch_helper.setdefault(pg, {})
            ch2 = ch1.setdefault(p, len(ch1) + 1)
            track[(pg, p, v)] = trk
            channel[(pg, p, v)] = ch2
        elif mode == 2:
            track[(pg, p, v)] = 0
            ch = ch_helper.setdefault(p, len(ch_helper) + 1)
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
            raise Exception("unsupported part/voice assign mode {}".format(mode))

    result = dict((k, (track.get(k, 0), channel.get(k, 1))) for k in note_keys)
    # for (pg, p, voice), v in result.items():
    #     pgn = pg.group_name if hasattr(pg, 'group_name') else pg.id
    #     print(pgn, p.id, voice)
    #     print(v)
    #     print()
    return result


def get_ppq(parts):
    ppqs = np.concatenate(
        [part.quarter_durations()[:, 1] for part in score.iter_parts(parts)]
    )
    ppq = np.lcm.reduce(ppqs)
    return ppq


@deprecated_alias(performed_part="performance_data")
def save_performance_midi(
    performance_data: PerformanceLike,
    out: Optional[PathLike],
    mpq: int = 500000,
    ppq: int = 480,
    default_velocity: int = 64,
    merge_tracks_save: Optional[bool] = False,
) -> Optional[MidiFile]:
    """Save a :class:`~partitura.performance.PerformedPart` or
    a :class:`~partitura.performance.Performance` as a MIDI file

    Parameters
    ----------
    performance_data : PerformanceLike
        The performance to be saved.
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
    merge_tracks_save : bool, optional
        Determines whether midi tracks are merged when exporting to a midi file. Defaults to False.

    Returns
    -------
    None or MidiFile
        If no output is specified using `out`, the function returns
        a `MidiFile` object. Otherwise, the function returns None.
    """

    if isinstance(performance_data, Performance):
        performed_parts = performance_data.performedparts
    elif isinstance(performance_data, PerformedPart):
        performed_parts = [performance_data]
    elif isinstance(performance_data, Iterable):
        if not all(isinstance(pp, PerformedPart) for pp in performance_data):
            raise ValueError(
                "`performance_data` should be a `Performance`, a `PerformedPart`,"
                " or a list of  `PerformedPart` instances"
            )
        performed_parts = performed_parts

    else:
        raise ValueError(
            "`performance_data` should be a `Performance`, a `PerformedPart`,"
            f" or a list of  `PerformedPart` instances but is {type(performance_data)}"
        )

    track_events = defaultdict(lambda: defaultdict(list))
    for performed_part in performed_parts:

        for c in performed_part.meta_other:
            track = c.get("track", 0)
            t = int(np.round(10**6 * ppq * c["time"] / mpq))
            msg_info = dict(
                [
                    (key, val)
                    for key, val in c.items()
                    if key not in ("time", "time_tick", "track")
                ]
            )
            track_events[track][t].append(MetaMessage(**msg_info))

        for c in performed_part.key_signatures:
            track = c.get("track", 0)
            t = int(np.round(10**6 * ppq * c["time"] / mpq))
            track_events[track][t].append(
                MetaMessage(
                    type="key_signature",
                    key=fifths_mode_to_key_name(
                        fifths=c.get("fifths", 0),
                        mode=c.get("mode", None),
                    ),
                )
            )

        for c in performed_part.time_signatures:
            track = c.get("track", 0)
            t = int(np.round(10**6 * ppq * c["time"] / mpq))
            track_events[track][t].append(
                MetaMessage(
                    type="time_signature",
                    numerator=c.get("beats", 4),
                    denominator=c.get("beat_type", 4),
                ),
            )

        for c in performed_part.controls:
            track = c.get("track", 0)
            ch = c.get("channel", 1)
            t = int(np.round(10**6 * ppq * c["time"] / mpq))
            track_events[track][t].append(
                Message(
                    "control_change",
                    control=c["number"],
                    value=c["value"],
                    channel=ch,
                )
            )

        for n in performed_part.notes:
            track = n.get("track", 0)
            ch = n.get("channel", 1)
            t_on = int(np.round(10**6 * ppq * n["note_on"] / mpq))
            t_off = int(np.round(10**6 * ppq * n["note_off"] / mpq))
            vel = n.get("velocity", default_velocity)
            track_events[track][t_on].append(
                Message("note_on", note=n["midi_pitch"], velocity=vel, channel=ch)
            )
            track_events[track][t_off].append(
                Message("note_off", note=n["midi_pitch"], velocity=0, channel=ch)
            )

        for p in performed_part.programs:
            track = p.get("track", 0)
            ch = p.get("channel", 1)
            t = int(np.round(10**6 * ppq * p["time"] / mpq))
            track_events[track][t].append(
                Message("program_change", program=int(p["program"]), channel=ch)
            )

        if len(performed_part.programs) == 0:
            # Add default program (to each track/channel)
            channels_and_tracks = np.array(
                list(
                    set(
                        [
                            (c.get("channel", 1), c.get("track", 0))
                            for c in performed_part.controls
                        ]
                        + [
                            (n.get("channel", 1), n.get("track", 0))
                            for n in performed_part.notes
                        ]
                    )
                ),
                dtype=int,
            )

            timepoints = []
            for tr in track_events.keys():
                timepoints += list(track_events[tr].keys())
            timepoints = list(set(timepoints))

            for tr in np.unique(channels_and_tracks[:, 1]):
                channel_idxs = np.where(channels_and_tracks[:, 1] == tr)[0]
                track_channels = np.unique(channels_and_tracks[channel_idxs, 0])
                for ch in track_channels:
                    track_events[tr][min(timepoints)].append(
                        Message("program_change", program=0, channel=ch)
                    )

    midi_type = 0 if len(track_events) == 1 else 1

    mf = MidiFile(type=midi_type, ticks_per_beat=ppq)

    for j, i in enumerate(sorted(track_events.keys())):
        track = MidiTrack()
        mf.tracks.append(track)
        if j == 0:
            track.append(MetaMessage("set_tempo", tempo=mpq, time=0))
        t = 0
        for t_msg in sorted(track_events[i].keys()):
            t_delta = t_msg - t
            for msg in track_events[i][t_msg]:
                track.append(msg.copy(time=t_delta))
                t_delta = 0
            t = t_msg

    if merge_tracks_save and len(mf.tracks) > 1:
        mf.tracks = [merge_tracks(mf.tracks)]

    if out is not None:
        if hasattr(out, "write"):
            mf.save(file=out)
        else:
            mf.save(out)
    else:
        return mf


@deprecated_alias(parts="score_data")
def save_score_midi(
    score_data: ScoreLike,
    out: Optional[PathLike],
    part_voice_assign_mode: int = 0,
    velocity: int = 64,
    anacrusis_behavior: str = "shift",
    minimum_ppq: int = 0,
) -> Optional[MidiFile]:
    """Write data from Part objects to a MIDI file

    Parameters
    ----------
    score_data : Score, list, Part, or PartGroup
        The musical score to be saved. A :class:`partitura.score.Score` object,
        a :class:`partitura.score.Part`, a :class:`partitura.score.PartGroup` or
        a list of these.
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

        The default mode is 0.
    velocity : int, optional
        Default velocity for all MIDI notes. Defaults to 64.
    anacrusis_behavior : {"shift", "pad_bar", "time_sig_change"}, optional
        Strategy to deal with anacrusis. If "shift", all
        time points are shifted by the anacrusis (i.e., the first
        note starts at 0). If "pad_bar", the "incomplete" bar  of
        the anacrusis is padded with silence. Defaults to 'shift'.
        If "time_sig_change", the time signature is changed to match
        the duration of the measure. This also ensure the beat and
        downbeats position are coherent in case of incomplete measures
        later in the score.
    minimum_ppq : int, optional
        Minimum ppq to use for the MIDI file. If the ppq of the score is less,
        it will be doubled until it is above the threshold. This is useful
        because some libraries like miditok require a certain minimum ppq to
        work properly.

    Returns
    -------
    None or MidiFile
        If no output is specified using `out`, the function returns
        a `MidiFile` object. Otherwise, the function returns None.
    """

    if isinstance(score_data, Score):
        parts = score_data.parts
    elif isinstance(score_data, (Part, PartGroup)):
        parts = [score_data]
    elif isinstance(score_data, Iterable):
        parts = score_data

    else:
        raise ValueError(
            "`score_data` should be a `Score`, a `Part`, a `PartGroup"
            f" or a list of  `Part` instances but is {type(score_data)}"
        )
    ppq = get_ppq(parts)
    # double it until it is above the minimum level.
    # Doubling instead of setting it ensure that the common divisors stay the same.
    while ppq < minimum_ppq:
        ppq = ppq * 2

    events = defaultdict(lambda: defaultdict(list))
    meta_events = defaultdict(lambda: defaultdict(list))

    event_keys = OrderedDict()
    tempos = {}

    quarter_maps = [part.quarter_map for part in score.iter_parts(parts)]

    first_time_point = min(qm(0) for qm in quarter_maps)

    ftp = 0
    # Deal with anacrusis
    if first_time_point < 0:
        if anacrusis_behavior == "shift" or anacrusis_behavior == "time_sig_change":
            ftp = first_time_point
        elif anacrusis_behavior == "pad_bar":
            time_signatures = []
            for qm, part in zip(quarter_maps, score.iter_parts(parts)):
                ts_beats, ts_beat_type, ts_mus_beats = part.time_signature_map(0)
                time_signatures.append((ts_beats, ts_beat_type, qm(0)))
            # sort ts according to time
            time_signatures.sort(key=lambda x: x[2])
            ftp = -time_signatures[0][0] / (time_signatures[0][1] / 4)
        else:
            raise Exception(
                'Invalid anacrusis_behavior value, must be one of ("shift", "pad_bar")'
            )

    for qm, part in zip(quarter_maps, score.iter_parts(parts)):
        pg = get_partgroup(part)

        notes = part.notes_tied

        def to_ppq(t):
            # convert div times to new ppq
            return int(ppq * (qm(t) - ftp))

        for tp in part.iter_all(score.Tempo):
            tempos[to_ppq(tp.start.t)] = MetaMessage(
                "set_tempo", tempo=tp.microseconds_per_quarter
            )
        # default tempo
        if not tempos:
            tempos[0] = MetaMessage("set_tempo", tempo=500000)

        if anacrusis_behavior == "time_sig_change":
            # Change time signature to match the duration of the measure
            # This ensure the beat and downbeats position are coherent
            # in case of incomplete measures later in the score.
            all_ts = list(part.iter_all(score.TimeSignature))
            ts_changing_time = [ts.start.t for ts in all_ts]
            for measure in part.iter_all(score.Measure):
                m_duration_beat = part.beat_map(measure.end.t) - part.beat_map(
                    measure.start.t
                )
                m_ts = part.time_signature_map(measure.start.t)
                if m_duration_beat != m_ts[0]:
                    # add ts change
                    # TODO: add support for changing the beat type if number of beats is not integer
                    meta_events[part][to_ppq(measure.start.t)].append(
                        MetaMessage(
                            "time_signature",
                            numerator=int(m_duration_beat),
                            denominator=int(m_ts[1]),
                        )
                    )
                    ts_changing_time.append(
                        measure.start.t
                    )  # keep track of changing the ts
                    # now go back to original ts if there is no ts change after this measure
                    if not any([ts_t > measure.start.t for ts_t in ts_changing_time]):
                        meta_events[part][to_ppq(measure.end.t)].append(
                            MetaMessage(
                                "time_signature",
                                numerator=int(m_ts[0]),
                                denominator=int(m_ts[1]),
                            )
                        )
            # filter out the multiple ts changes at the same time
            # this happens when multiple measure in a row have wrong duration
            for t in meta_events[part].keys():
                if len(meta_events[part][t]) == 2:
                    meta_events[part][t] = meta_events[part][t][1:]

            # now add the normal time signature change
            for ts in part.iter_all(score.TimeSignature):
                if ts.start.t in ts_changing_time:
                    # don't add if something is already added at this time to cover the case of a ts change when the first measure is shorter/longer
                    pass
                else:
                    meta_events[part][to_ppq(ts.start.t)].append(
                        MetaMessage(
                            "time_signature",
                            numerator=ts.beats,
                            denominator=ts.beat_type,
                        )
                    )
        else:  # just add the time signature that are explicit in partitura
            for i, ts in enumerate(part.iter_all(score.TimeSignature)):
                if anacrusis_behavior == "pad_bar" and i == 0:
                    # shift the first time signature to 0 so MIDI players can pick up the correct measure position
                    meta_events[part][0].append(
                        MetaMessage(
                            "time_signature",
                            numerator=ts.beats,
                            denominator=ts.beat_type,
                        )
                    )
                else:  # follow the position in the partitura part
                    meta_events[part][to_ppq(ts.start.t)].append(
                        MetaMessage(
                            "time_signature",
                            numerator=ts.beats,
                            denominator=ts.beat_type,
                        )
                    )

        for ks in part.iter_all(score.KeySignature):
            meta_events[part][to_ppq(ks.start.t)].append(
                MetaMessage("key_signature", key=ks.name)
            )

        for note in notes:
            # key is a tuple (part_group, part, voice) that will be
            # converted into a (track, channel) pair.
            key = (pg, part, note.voice)
            events[key][to_ppq(note.start.t)].append(
                Message("note_on", note=note.midi_pitch)
            )
            events[key][to_ppq(note.start.t + note.duration_tied)].append(
                Message("note_off", note=note.midi_pitch)
            )
            event_keys[key] = True

    tr_ch_map = map_to_track_channel(list(event_keys.keys()), part_voice_assign_mode)

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
        if hasattr(out, "write"):
            mf.save(file=out)
        else:
            mf.save(out)
    else:
        return mf
