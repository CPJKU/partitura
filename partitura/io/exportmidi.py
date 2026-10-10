#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for exporting MIDI files
"""

import io
import warnings

import numpy as np

from collections import defaultdict, OrderedDict
from typing import Optional, Iterable

from mido import MidiFile, MidiTrack, Message, MetaMessage, merge_tracks

try:
    from partitura import _minimidi

    _HAS_FASTMIDI = True
except ImportError:
    _minimidi = None
    _HAS_FASTMIDI = False

# SMF type-1 / type-0 headers store ticks-per-beat in a signed 16-bit field,
# so the largest representable ppq is 32767. Without a cap, scores with many
# coprime divisions (e.g., 3:2 and 5:4 and 7:6 tuplets in the same piece) make
# np.lcm.reduce explode past this limit and crash mido at write time with
# `struct.error: 'h' format requires -32768 <= number <= 32767`.
SMF_MAX_PPQ = 32767

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
            ch2 = ch1.setdefault(v, len(ch1))
            track[(pg, p, v)] = trk
            channel[(pg, p, v)] = ch2
        elif mode == 1:
            trk = tr_helper.setdefault(pg, len(tr_helper))
            ch1 = ch_helper.setdefault(pg, {})
            ch2 = ch1.setdefault(p, len(ch1))
            track[(pg, p, v)] = trk
            channel[(pg, p, v)] = ch2
        elif mode == 2:
            track[(pg, p, v)] = 0
            ch = ch_helper.setdefault(p, len(ch_helper))
            channel[(pg, p, v)] = ch
        elif mode == 3:
            trk = tr_helper.setdefault(p, len(tr_helper))
            track[(pg, p, v)] = trk
            channel[(pg, p, v)] = 0
        elif mode == 4:
            track[(pg, p, v)] = 0
            channel[(pg, p, v)] = 0
        elif mode == 5:
            trk = tr_helper.setdefault((p, v), len(tr_helper))
            track[(pg, p, v)] = trk
            channel[(pg, p, v)] = 0
        else:
            raise Exception("unsupported part/voice assign mode {}".format(mode))

    result = dict((k, (track.get(k, 0), channel.get(k, 0))) for k in note_keys)
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
    ppq = int(np.lcm.reduce(ppqs))
    if ppq > SMF_MAX_PPQ:
        warnings.warn(
            f"Computed ppq={ppq} exceeds the SMF 16-bit max ({SMF_MAX_PPQ}); "
            f"capping. Tick positions for very-fine subdivisions may be rounded.",
            RuntimeWarning,
            stacklevel=2,
        )
        ppq = SMF_MAX_PPQ
    return ppq


# Must match `EventRecord` / `EventKind` in native/fastmidi.cpp.
_EVENT_DTYPE = np.dtype(
    [
        ("tick", "<i8"),
        ("kind", "<i4"),
        ("channel", "<i4"),
        ("a", "<i4"),
        ("b", "<i4"),
    ]
)
(
    _NOTE_ON,
    _NOTE_OFF,
    _CONTROL,
    _PROGRAM,
    _TIME_SIG,
    _KEY_SIG,
    _RAW_META,
) = range(7)


def _check_range(name, values, lo, hi):
    # mido validates these when building each Message; the raw minimidi
    # writer does not, so check up front (vectorized) and raise the same
    # kind of error.
    values = np.asarray(values)
    if values.size and (values.min() < lo or values.max() > hi):
        raise ValueError(f"{name} must be between {lo} and {hi}")


class _EventCollector:
    """Accumulates tick-converted events per output track, in insertion
    order, mirroring the `track_events[track][tick]` bookkeeping of the
    mido-based exporter (events sharing a tick keep their insertion order
    because the writer stable-sorts by tick).
    """

    def __init__(self):
        self.blocks = defaultdict(list)
        self.raw_metas = []
        self.min_tick = None

    def add(self, tracks, ticks, kind, channel, a, b):
        ticks = np.asarray(ticks, dtype=np.int64)
        if ticks.size == 0:
            return
        if ticks.min() < 0:
            raise ValueError("MIDI event times must be non-negative")
        events = np.empty(ticks.size, dtype=_EVENT_DTYPE)
        events["tick"] = ticks
        events["kind"] = kind
        events["channel"] = channel
        events["a"] = a
        events["b"] = b
        tracks = np.asarray(tracks, dtype=np.int64)
        tmin = int(ticks.min())
        self.min_tick = tmin if self.min_tick is None else min(self.min_tick, tmin)
        if (tracks == tracks[0]).all():
            # common case (single track): skip sorting/splitting by track
            self.blocks[int(tracks[0])].append(events)
            return
        for tr in np.unique(tracks):
            self.blocks[int(tr)].append(events[tracks == tr])


def _seconds_to_ticks(times, ppq, mpq):
    # same arithmetic (and operation order) as the mido-based exporter's
    # int(np.round(10**6 * ppq * t / mpq)), so ticks are identical
    return np.round((10**6 * ppq) * np.asarray(times, dtype=float) / mpq).astype(
        np.int64
    )


def _build_midi_bytes_minimidi(
    performed_parts, mpq, ppq, default_velocity, merge_tracks_save
):
    """Serialize `performed_parts` to the bytes of a Standard MIDI File
    using the compiled `_minimidi` extension. Mirrors the event layout and
    ordering of the mido-based path in `save_performance_midi`.
    """
    if not 0 < ppq <= SMF_MAX_PPQ:
        raise ValueError(f"ppq must be between 1 and {SMF_MAX_PPQ}")

    col = _EventCollector()

    for pp in performed_parts:
        # --- meta_other: arbitrary meta messages, encoded by mido and
        # passed through as raw [type, length, payload...] bytes
        if pp.meta_other:
            raw_idx = []
            for c in pp.meta_other:
                msg_info = {
                    k: v for k, v in c.items() if k not in ("time", "time_tick", "track")
                }
                raw_idx.append(len(col.raw_metas))
                col.raw_metas.append(bytes(MetaMessage(**msg_info).bytes()[1:]))
            col.add(
                [c.get("track", 0) for c in pp.meta_other],
                _seconds_to_ticks([c["time"] for c in pp.meta_other], ppq, mpq),
                _RAW_META,
                0,
                raw_idx,
                0,
            )

        # --- key signatures
        if pp.key_signatures:
            fifths, minor = [], []
            for c in pp.key_signatures:
                f, mode = c.get("fifths", 0), c.get("mode", None)
                # validates fifths/mode exactly like the mido-based path
                fifths_mode_to_key_name(fifths=f, mode=mode)
                fifths.append(f)
                minor.append(1 if mode in ("minor", -1) else 0)
            col.add(
                [c.get("track", 0) for c in pp.key_signatures],
                _seconds_to_ticks([c["time"] for c in pp.key_signatures], ppq, mpq),
                _KEY_SIG,
                0,
                fifths,
                minor,
            )

        # --- time signatures
        if pp.time_signatures:
            beats = np.array([c.get("beats", 4) for c in pp.time_signatures])
            beat_type = np.array([c.get("beat_type", 4) for c in pp.time_signatures])
            _check_range("time signature numerator", beats, 0, 255)
            if np.any(beat_type < 1) or np.any(beat_type & (beat_type - 1)):
                raise ValueError("time signature denominator must be a power of two")
            col.add(
                [c.get("track", 0) for c in pp.time_signatures],
                _seconds_to_ticks([c["time"] for c in pp.time_signatures], ppq, mpq),
                _TIME_SIG,
                0,
                beats,
                beat_type,
            )

        # (track, channel) pairs of this part's notes and controls, for the
        # default program_change insertion below
        track_channel = []

        # --- control changes
        if pp.controls:
            tracks = np.array([c.get("track", 0) for c in pp.controls])
            channels = np.array([c.get("channel", 0) for c in pp.controls])
            numbers = np.array([c["number"] for c in pp.controls])
            values = np.array([c["value"] for c in pp.controls])
            _check_range("control number", numbers, 0, 127)
            _check_range("control value", values, 0, 127)
            _check_range("channel", channels, 0, 15)
            col.add(
                tracks,
                _seconds_to_ticks([c["time"] for c in pp.controls], ppq, mpq),
                _CONTROL,
                channels,
                numbers,
                values,
            )
            track_channel.append(np.column_stack((tracks, channels)))

        # --- notes (note_on / note_off interleaved per note, as in the
        # mido-based path)
        if pp.notes:
            rows = [
                (
                    d["midi_pitch"],
                    d["note_on"],
                    d["note_off"],
                    d.get("velocity", default_velocity),
                    d.get("channel", 0),
                    d.get("track", 0),
                )
                for d in (n.pnote_dict for n in pp.notes)
            ]
            arr = np.array(rows, dtype=float)
            pitch, vel, channels, tracks = (
                arr[:, 0].astype(np.int64),
                arr[:, 3].astype(np.int64),
                arr[:, 4].astype(np.int64),
                arr[:, 5].astype(np.int64),
            )
            _check_range("note pitch", pitch, 0, 127)
            _check_range("note velocity", vel, 0, 127)
            _check_range("channel", channels, 0, 15)

            n_notes = len(rows)
            ticks = np.empty(2 * n_notes, dtype=np.int64)
            ticks[0::2] = _seconds_to_ticks(arr[:, 1], ppq, mpq)
            ticks[1::2] = _seconds_to_ticks(arr[:, 2], ppq, mpq)
            kinds = np.tile(np.array([_NOTE_ON, _NOTE_OFF]), n_notes)
            velocities = np.zeros(2 * n_notes, dtype=np.int64)
            velocities[0::2] = vel
            col.add(
                np.repeat(tracks, 2),
                ticks,
                kinds,
                np.repeat(channels, 2),
                np.repeat(pitch, 2),
                velocities,
            )
            track_channel.append(np.column_stack((tracks, channels)))

        # --- program changes
        if pp.programs:
            channels = np.array([p.get("channel", 0) for p in pp.programs])
            programs = np.array([int(p["program"]) for p in pp.programs])
            _check_range("program", programs, 0, 127)
            _check_range("channel", channels, 0, 15)
            col.add(
                [p.get("track", 0) for p in pp.programs],
                _seconds_to_ticks([p["time"] for p in pp.programs], ppq, mpq),
                _PROGRAM,
                channels,
                programs,
                0,
            )
        elif track_channel:
            # default program (to each track/channel), placed at the earliest
            # tick of any event written so far, after the events already there
            pairs = np.unique(np.concatenate(track_channel), axis=0)
            col.add(
                pairs[:, 0],
                np.full(len(pairs), col.min_tick),
                _PROGRAM,
                pairs[:, 1],
                0,
                0,
            )

    track_arrays = [np.concatenate(col.blocks[i]) for i in sorted(col.blocks)]
    midi_type = 0 if len(track_arrays) == 1 else 1
    if merge_tracks_save and len(track_arrays) > 1:
        track_arrays = [np.concatenate(track_arrays)]

    return _minimidi.write_midi(track_arrays, col.raw_metas, ppq, mpq, midi_type)


@deprecated_alias(performed_part="performance_data")
def save_performance_midi(
    performance_data: PerformanceLike,
    out: Optional[PathLike],
    mpq: int = 500000,
    ppq: int = 480,
    default_velocity: int = 64,
    merge_tracks_save: Optional[bool] = False,
    backend: str = "auto",
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
    backend : {'auto', 'mido', 'minimidi'}, optional
        Which library serializes the MIDI file. ``'mido'`` builds
        `mido` messages and is the original implementation.
        ``'minimidi'`` uses the optional, much faster compiled
        extension (the same one as `load_performance_midi`'s
        ``'minimidi'`` backend) and raises a `RuntimeError` if it is
        not available. ``'auto'`` (the default) uses the compiled
        extension when available and `mido` otherwise. Both backends
        write the same events in the same order; the files are not
        byte-identical (minimidi uses running status and always writes
        its end-of-track marker one tick after the last event), but
        read back to the same notes, controls, programs and
        signatures.

    Returns
    -------
    None or MidiFile
        If no output is specified using `out`, the function returns
        a `MidiFile` object. Otherwise, the function returns None.
    """
    if backend not in ("auto", "mido", "minimidi"):
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of 'auto', 'mido', 'minimidi'"
        )
    if backend == "minimidi" and not _HAS_FASTMIDI:
        raise RuntimeError(
            "backend='minimidi' was requested but the optional "
            "partitura._minimidi extension is not built. Use "
            "backend='mido' (or 'auto'), or reinstall partitura with "
            "a C++ toolchain available."
        )
    use_fast = backend == "minimidi" or (backend == "auto" and _HAS_FASTMIDI)

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

    if use_fast:
        data = _build_midi_bytes_minimidi(
            performed_parts, mpq, ppq, default_velocity, merge_tracks_save
        )
        if out is None:
            # keep the return type of the mido-based path
            return MidiFile(file=io.BytesIO(data))
        if hasattr(out, "write"):
            out.write(data)
        else:
            with open(out, "wb") as f:
                f.write(data)
        return

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
            ch = c.get("channel", 0)
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
            ch = n.get("channel", 0)
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
            ch = p.get("channel", 0)
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
                            (c.get("channel", 0), c.get("track", 0))
                            for c in performed_part.controls
                        ]
                        + [
                            (n.get("channel", 0), n.get("track", 0))
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
        # Materialize so the empty-check below doesn't consume a generator
        # that get_ppq / the main loop need to iterate again.
        parts = list(score_data)

    else:
        raise ValueError(
            "`score_data` should be a `Score`, a `Part`, a `PartGroup"
            f" or a list of  `Part` instances but is {type(score_data)}"
        )

    if not list(score.iter_parts(parts)):
        raise ValueError(
            "Cannot save to MIDI: the score has no parts. This usually "
            "means the source MusicXML had an empty <part-list/> or no "
            "<part> elements."
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
            return round(ppq * (qm(t) - ftp))

        for tp in part.iter_all(score.Tempo):
            tempos[to_ppq(tp.start.t)] = MetaMessage(
                "set_tempo", tempo=tp.microseconds_per_quarter
            )
        # default tempo
        if not tempos:
            tempos[0] = MetaMessage("set_tempo", tempo=500000)

        # note velocities
        velocities = []
        for vp in part.iter_all(score.Dynamic):
            velocities.append((vp.start.t, round(90 * vp.velocity / 100)))
        velocities = np.stack(velocities) if velocities else velocities

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
            vel = velocity
            if len(velocities) > 0:
                vel_idx = (
                    np.searchsorted(velocities[:, 0], note.start.t, side="right") - 1
                )
                vel = int(velocities[vel_idx, 1])
            events[key][to_ppq(note.start.t)].append(
                Message("note_on", note=note.midi_pitch, velocity=vel)
            )
            events[key][to_ppq(note.start.t + note.duration_tied)].append(
                Message("note_off", note=note.midi_pitch, velocity=vel)
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
