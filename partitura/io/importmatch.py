#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains methods for parsing matchfiles
"""

import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional, Callable, List
import warnings
from functools import partial
import numpy as np

from partitura import score
from partitura.score import Part, Score
from partitura.performance import PerformedPart, Performance

from partitura.io.matchlines_v0 import (
    FROM_MATCHLINE_METHODS as FROM_MATCHLINE_METHODSV0,
    MatchInfo as MatchInfoV0,
    MatchTrillNote as MatchTrillNoteV0,
)

from partitura.io.matchlines_v1 import (
    FROM_MATCHLINE_METHODS as FROM_MATCHLINE_METHODSV1,
    MatchInfo as MatchInfoV1,
    MatchOrnamentNote as MatchOrnamentNoteV1,
)

from partitura.io.matchfile_base import (
    MatchError,
    MatchFile,
    MatchLine,
    BaseSnoteLine,
    BaseSnoteNoteLine,
    BaseDeletionLine,
    BaseInsertionLine,
    BaseOrnamentLine,
    BaseOmittedSectionLine,
    VirtualSnoteLine,
    VirtualNoteLine,
    note_pair_classes,
)

from partitura.io.matchfile_utils import (
    Version,
    number_pattern,
    vnumber_pattern,
    fingering_pattern,
    MatchTimeSignature,
    format_pnote_id,
)

from partitura.utils.music import midi_ticks_to_seconds


from partitura.utils.misc import deprecated_alias, PathLike, get_document_name

from partitura.utils.generic import interp1d, iter_current_next

__all__ = ["load_match", "AlignmentInfo"]


@dataclass
class AlignmentInfo:
    """
    Information about an alignment loaded from a match file, in addition to
    the alignment itself.

    Attributes
    ----------
    note_types : List[Tuple[str, str]]
        The types of the notes for each element of the alignment, e.g.,
        ("snote", "note") or ("virtualSnote", "note") (see
        `note_alignment_info_from_matchfile`).
    score_attributes : List[List[str]]
        The score attributes for each element of the alignment.
    sections : List[dict]
        The sections in the match file (see `sections_from_matchfile`).
    omitted_sections : List[dict]
        The omitted sections in the match file.
    """

    note_types: List[Tuple[str, str]] = field(default_factory=list)
    score_attributes: List[List[str]] = field(default_factory=list)
    sections: List[dict] = field(default_factory=list)
    omitted_sections: List[dict] = field(default_factory=list)


def get_version(line: str) -> Version:
    """
    Get version from the first line. Since the
    first version of the format did not include this line,
    we assume that the version is 0.1.0 if no version is
    found.

    Parameters
    ----------
    line: str
        The first line of the match file.

    Returns
    -------
    version : Version
        The version of the match file
    """
    version = Version(0, 1, 0)

    for parser in (MatchInfoV1, MatchInfoV0):
        try:
            ml = parser.from_matchline(line)
            if isinstance(getattr(ml, "Value", None), Version):
                version = ml.Value
                return version

        except MatchError:
            pass

    return version


def parse_matchline(
    line: str,
    from_matchline_methods: List[Callable[[str], MatchLine]],
    version: Version,
    debug: bool = False,
) -> Optional[MatchLine]:
    """
    Return objects representing the line as one of:

    * hammer_bounce-PlayedNote.
    * info(Attribute, Value).
    * insertion-PlayedNote.
    * ornament(Anchor)-PlayedNote.
    * ScoreNote-deletion.
    * ScoreNote-PlayedNote.
    * ScoreNote-trailing_score_note.
    * ScoreNote-VirtualPlayedNote.
    * VirtualScoreNote-PlayedNote.
    * VirtualScoreNote-VirtualPlayedNote.
    * trailing_played_note-PlayedNote.
    * trill(Anchor)-PlayedNote.
    * meta(Attribute, Value, Bar, Beat).
    * section(...).
    * omittedSection(...).
    * sustain(Time, Value)
    * soft(Time, Value)

    or None if none can be matched

    Parameters
    ----------
    line : str
        Line of the match file
    from_matchline_methods : List[Callable[[str], MatchLine]]

    Returns
    -------
    matchline : subclass of `MatchLine`
       Object representing the line.
    """

    matchline = None
    for from_matchline in from_matchline_methods:
        try:
            matchline = from_matchline(line, version=version)
            break
        except Exception as e:
            if not isinstance(e, MatchError):
                print(line, e, version)  # pragma: no cover
            continue

    return matchline


@deprecated_alias(fn="filename", create_part="create_score")
def load_matchfile(
    filename: PathLike,
) -> MatchFile:
    """
    Load a Matchfile as a `MatchFile` instance
    """

    if not os.path.exists(filename):
        raise ValueError("Filename does not exist")  # pragma: no cover

    with open(filename) as f:
        raw_lines = f.read().splitlines()

    version = get_version(raw_lines[0])

    from_matchline_methods = FROM_MATCHLINE_METHODSV1
    if version < Version(1, 0, 0):
        from_matchline_methods = FROM_MATCHLINE_METHODSV0

    parsed_lines = list()
    # Functionality to remove duplicate lines
    len_raw_lines = len(raw_lines)
    np_lines = np.array(raw_lines, dtype=str)
    # Remove empty lines
    np_lines = np_lines[np_lines != ""]
    # Remove duplicate lines
    _, idx = np.unique(np_lines, return_index=True)
    np_lines = np_lines[np.sort(idx)]
    # Parse lines
    f = partial(
        parse_matchline, version=version, from_matchline_methods=from_matchline_methods
    )
    f_vec = np.vectorize(f)
    parsed_lines_raw = f_vec(np_lines)
    # do not return unparseable lines
    parsed_lines = parsed_lines_raw[parsed_lines_raw != None].tolist()
    # Create MatchFile instance
    mf = MatchFile(lines=parsed_lines)
    # Validate match for duplicate snote_ids or pnote_ids
    validate_match_ids(mf)
    return mf


@deprecated_alias(fn="filename", create_part="create_score")
def load_match(
    filename: PathLike,
    create_score: bool = False,
    pedal_threshold: int = 64,
    first_note_at_zero: bool = False,
    clip_velocities: bool = True,
    offset_duration_whole: bool = True,
    quiet: bool = False,
    return_alignment_info: bool = False,
) -> Tuple[Union[Performance, list, Score, AlignmentInfo]]:
    """
    Load a matchfile.

    Parameters
    ----------
    filename : str
        The matchfile
    create_score : bool, optional
        When True create a Part object from the snote information in
        the match file. Defaults to False.
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.
        Defaults to False.
    clip_velocities : bool, optional
        When True the velocity values are clipped to a range of 0 - 127.
        Defaults to True.
    offset_duration_whole: Boolean, optional
        A flag for the type of offset and duration given in the matchfile.
        When true, the function expects the values to be given in whole
        notes (e.g. 1/4 for a quarter note) independet of time signature.
        Defaults to True.
    quiet : bool, optional
        If True, suppress all warnings emitted during parsing.
        Defaults to False.
    return_alignment_info : bool, optional
        When True also return the types and the score attributes of the notes
        in the alignment, and the sections and omitted sections in the match file.
        Defaults to False.

    Returns
    -------
    performance : :class:partitura.performance.Performance
    alignment : list
        The score--performance alignment, a list of dictionaries
    scr : :class:partitura.score.Score
        The score. This item is only returned when `create_score` = True.
    alignment_info : AlignmentInfo
        The types and the score attributes of the notes for each element of
        the alignment, and the sections and omitted sections (see
        `AlignmentInfo`). This item is only returned when
        `return_alignment_info` = True.
    """
    ctx = warnings.catch_warnings() if quiet else nullcontext()
    with ctx:
        if quiet:
            warnings.simplefilter("ignore")

        # Parse Matchfile
        mf = load_matchfile(filename)

        # Generate PerformedPart
        ppart = performed_part_from_match(
            mf, pedal_threshold, first_note_at_zero, clip_velocities
        )

        performance = Performance(
            id=get_document_name(filename),
            performedparts=ppart,
            ensure_unique_tracks=False,
        )
        # Generate Part
        if create_score:
            spart = part_from_matchfile(
                mf,
                match_offset_duration_in_whole=offset_duration_whole,
            )

            scr = score.Score(id=get_document_name(filename), partlist=[spart])
        # Alignment
        alignment, note_types, score_attributes = note_alignment_info_from_matchfile(mf)

        result = (performance, alignment)

        if create_score:
            result += (scr,)

        if return_alignment_info:
            sections, omitted_sections = sections_from_matchfile(mf, first_note_at_zero)
            result += (
                AlignmentInfo(
                    note_types=note_types,
                    score_attributes=score_attributes,
                    sections=sections,
                    omitted_sections=omitted_sections,
                ),
            )

        return result


def note_alignment_from_matchfile(mf: MatchFile) -> List[dict]:
    """
    Get a note-level alignment from a MatchFile instance

    Parameters
    ----------
    mf : MatchFile
        A score-to-performance alignment

    Returns
    -------
    results : List[dict]
        An alignmnet as a list of dictionaries for each note.

    Notes
    -----
    * Alignments can be non-unique (from version 1.1.0 on), i.e., there can be
      several matches with the same `score_id` and/or `performance_id`. See
      `note_alignment_info_from_matchfile` for telling them apart.
    """
    alignment, _, _ = note_alignment_info_from_matchfile(mf)

    return alignment


# alias
alignment_from_matchfile = note_alignment_from_matchfile


def note_alignment_info_from_matchfile(
    mf: MatchFile,
) -> Tuple[List[dict], List[Tuple[str, str]], List[List[str]]]:
    """
    Get a note-level alignment from a MatchFile instance, and the types and
    the score attributes of the notes in each match line.

    Parameters
    ----------
    mf : MatchFile
        A score-to-performance alignment

    Returns
    -------
    alignment : List[dict]
        An alignment as a list of dictionaries for each note, in the order
        of the lines in the match file (which is sorted by performance time
        by default, see `save_match`).
    note_types : List[Tuple[str, str]]
        The types of the score note and the performance note for each element
        of the alignment:

        * `("snote", "note")`, `("virtualSnote", "note")`,
          `("snote", "virtualPnote")` or `("virtualSnote", "virtualPnote")`
          for matches
        * `("snote", "-")` for deletions
        * `("-", "note")` for insertions
        * `("ornament", "note")` for ornaments
    score_attributes : List[List[str]]
        The attributes of the score note (or virtual score note) for each
        element of the alignment. The list is empty if there are none.
    """
    alignment = []
    note_types = []
    score_attributes = []

    for line in mf.lines:
        if isinstance(line, note_pair_classes):
            al_note = dict(
                label="match",
                score_id=str(line.snote.Anchor),
                performance_id=format_pnote_id(line.note.Id),
            )
            is_virtual_snote = isinstance(line.snote, VirtualSnoteLine)
            is_virtual_pnote = isinstance(line.note, VirtualNoteLine)
            note_type = (
                "virtualSnote" if is_virtual_snote else "snote",
                "virtualPnote" if is_virtual_pnote else "note",
            )
            attributes = (
                line.snote.AttributesList
                if is_virtual_snote
                else line.snote.ScoreAttributesList
            )

        elif isinstance(line, BaseDeletionLine):
            if "leftOutTied" in line.snote.ScoreAttributesList:
                continue
            al_note = dict(label="deletion", score_id=str(line.snote.Anchor))
            note_type = ("snote", "-")
            attributes = line.snote.ScoreAttributesList

        elif isinstance(line, BaseInsertionLine):
            al_note = dict(
                label="insertion", performance_id=format_pnote_id(line.note.Id)
            )
            note_type = ("-", "note")
            attributes = []

        elif isinstance(line, BaseOrnamentLine):
            if isinstance(line, MatchTrillNoteV0):
                ornament_type = "trill"
            elif isinstance(line, MatchOrnamentNoteV1):
                ornament_type = line.OrnamentType
            else:
                ornament_type = "generic_ornament"
            al_note = dict(
                label="ornament",
                score_id=str(line.Anchor),
                performance_id=format_pnote_id(line.note.Id),
                type=ornament_type,
            )
            note_type = ("ornament", "note")
            attributes = []

        else:
            continue

        alignment.append(al_note)
        note_types.append(note_type)
        # empty lists in the match file are parsed as [""]
        score_attributes.append([a for a in attributes if a != ""])

    return alignment, note_types, score_attributes


def sections_from_matchfile(
    mf: MatchFile,
    first_note_at_zero: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """
    Get the sections and the omitted sections in a MatchFile instance, in the
    format used by `save_match`.

    Parameters
    ----------
    mf : MatchFile
        A MatchFile instance
    first_note_at_zero : bool, optional
        When True the performance times are shifted to make the first
        note_on time equal zero (see `performed_part_from_match`).

    Returns
    -------
    sections : List[dict]
        A list of dictionaries for each section. For match files before version
        1.1.0, `id`, `start_in_perf_time` and `end_in_perf_time` are None and
        `section_attr_list` is the type of the repeat end.
    omitted_sections : List[dict]
        A list of dictionaries for each omitted section.
    """
    offset_tick = 0
    if first_note_at_zero and len(mf.notes) > 0:
        offset_tick = max(0, min(note.Onset for note in mf.notes))

    sections, omitted_sections = [], []
    for line in mf.sections + mf.omitted_sections:
        section = dict(
            id=getattr(line, "id", None),
            start_in_beats_unfolded=line.StartInBeatsUnfolded,
            end_in_beats_unfolded=line.EndInBeatsUnfolded,
            start_in_beats_original=line.StartInBeatsOriginal,
            end_in_beats_original=line.EndInBeatsOriginal,
        )
        if isinstance(line, BaseOmittedSectionLine):
            omitted_sections.append(section)
        else:
            start, end = None, None
            if line.version >= Version(1, 1, 0):
                start = line.StartInPerfTime - offset_tick
                end = line.EndInPerfTime - offset_tick
            section.update(start_in_perf_time=start, end_in_perf_time=end)
            sections.append(section)

        # empty lists in the match file are parsed as [""]
        attributes = getattr(
            line, "SectionAttrList", getattr(line, "RepeatEndType", [])
        )
        section["section_attr_list"] = [a for a in attributes if a != ""]

    return sections, omitted_sections


def performed_part_from_match(
    mf: MatchFile,
    pedal_threshold: int = 64,
    first_note_at_zero: bool = False,
    clip_velocities: bool = True,
) -> PerformedPart:
    """
    Make PerformedPart from performance info in a MatchFile

    Parameters
    ----------
    mf : MatchFile
        A MatchFile instance
    pedal_threshold : int, optional
        Threshold for adjusting sound off of the performed notes using
        pedal information. Defaults to 64.
    first_note_at_zero : bool, optional
        When True the note_on and note_off times in the performance
        are shifted to make the first note_on time equal zero.
    clip_velocities : bool, optional
        When True the velocity values are clipped to a range of 0 - 127.
        Defaults to True.

    Returns
    -------
    ppart : PerformedPart
        A performed part

    """
    # Get midi time units
    mpq = mf.info("midiClockRate")  # 500000 -> microseconds per quarter
    ppq = mf.info("midiClockUnits")  # 500 -> parts per quarter

    # PerformedNote instances for all MatchNotes
    notes = []

    notes = list()
    note_onsets_in_secs = np.array(np.zeros(len(mf.notes)), dtype=float)
    note_onsets_in_tick = np.array(np.zeros(len(mf.notes)), dtype=int)
    tracks = set()
    channels = set()
    for i, note in enumerate(mf.notes):
        n_onset_sec = midi_ticks_to_seconds(note.Onset, mpq, ppq)
        note_onsets_in_secs[i] = n_onset_sec
        note_onsets_in_tick[i] = note.Onset
        track = getattr(note, "Track", 0)
        tracks.add(track)
        channel = getattr(note, "Channel", 0)
        channels.add(channel)
        velocity = note.Velocity
        if clip_velocities and (velocity > 127 or velocity < 0):
            warnings.warn(f"Clipping invalid velocity value ({velocity}).")
            velocity = np.clip(velocity, 0, 127)
        notes.append(
            dict(
                id=format_pnote_id(note.Id),
                midi_pitch=note.MidiPitch,
                note_on=n_onset_sec,
                note_off=midi_ticks_to_seconds(note.Offset, mpq, ppq),
                note_on_tick=note.Onset,
                note_off_tick=note.Offset,
                sound_off=midi_ticks_to_seconds(note.Offset, mpq, ppq),
                velocity=velocity,
                track=track,
                channel=channel,
            )
        )
    # Set first note_on to zero in ticks and seconds if first_note_at_zero
    if first_note_at_zero and len(note_onsets_in_secs) > 0:
        offset = note_onsets_in_secs.min()
        offset_tick = note_onsets_in_tick.min()
        if offset > 0 and offset_tick > 0:
            for note in notes:
                note["note_on"] -= offset
                note["note_off"] -= offset
                note["sound_off"] -= offset
                note["note_on_tick"] -= offset_tick
                note["note_off_tick"] -= offset_tick

    # check if multiple tracks are in the match file
    if len(tracks) > 1:
        warnings.warn("Notes on multiple MIDI tracks in matchfile" "information!.")
    used_track = min(tracks)
    # check if multiple tracks are in the match file
    if len(channels) > 1:
        warnings.warn("Notes on multiple MIDI channels in matchfile" "information!.")
    used_channel = min(channels)

    # SustainPedal instances for sustain pedal lines
    sustain_pedal = [
        dict(
            number=64,
            time=midi_ticks_to_seconds(ped.Time, mpq, ppq),
            value=ped.Value,
            track=used_track,
            channel=used_channel,
        )
        for ped in mf.sustain_pedal
    ]

    # SoftPedal instances for soft pedal lines
    soft_pedal = [
        dict(
            number=67,
            time=midi_ticks_to_seconds(ped.Time, mpq, ppq),
            value=ped.Value,
            track=used_track,
            channel=used_channel,
        )
        for ped in mf.soft_pedal
    ]

    # Make performed part
    ppart = PerformedPart(
        id="P1",
        part_name=mf.info("piece"),
        notes=notes,
        controls=sustain_pedal + soft_pedal,
        sustain_pedal_threshold=pedal_threshold,
        ppq=ppq,
        mpq=mpq,
        track=used_track,
    )
    return ppart


def sort_snotes(snotes: List[BaseSnoteLine]) -> List[BaseSnoteLine]:
    """
    Sort s(core)notes.

    Parameters
    ----------
    snotes : list
        The score notes

    Returns
    -------
    snotes_sorted : list
        The sorted score notes
    """
    sidx = np.lexsort(
        list(zip(*[(float(n.Offset), float(n.Beat), float(n.Measure)) for n in snotes]))
    )
    return [snotes[i] for i in sidx if snotes[i].NoteName.lower() != "r"]


def part_from_matchfile(
    mf: MatchFile,
    match_offset_duration_in_whole: bool = True,
) -> Part:
    """
    Create a score part from a matchfile.

    Parameters
    ----------
    mf : MatchFile
        An instance of `MatchFile`

    match_offset_duration_in_whole: Boolean
        A flag for the type of offset and duration given in the matchfile.
        When true, the function expects the values to be given in whole
        notes (e.g. 1/4 for a quarter note) independet of time signature.


    Returns
    -------
    part : partitura.score.Part
        An instance of `Part` containing score information.

    Notes
    -----
    * The score notes in omitted sections are not listed in the match file, so
      the measures in these sections are empty (unless they contain notes that
      are aligned, e.g., by means of virtual notes). The omitted sections are
      represented as `OmittedSection` objects in the part.
    """
    part = score.Part("P1", mf.info("piece"))
    snotes = sort_snotes(mf.snotes)

    ts = mf.time_signatures
    omitted_sections = mf.omitted_sections
    min_time = min(n.OnsetInBeats for n in snotes)
    max_time = max(
        [n.OffsetInBeats for n in snotes]
        + [s.EndInBeatsUnfolded for s in omitted_sections]
        + [ts_beat_time for ts_beat_time, _, _ in ts]
    )
    beats_to_quarters = make_beats_to_quarters_map(ts)
    (
        beats_map_from_beats,
        beats_map_from_quarters,
        beat_type_map_from_beats,
        beat_type_map_from_quarters,
        min_time_q,
        max_time_q,
    ) = make_timesig_maps(ts, max_time)

    # compute necessary divs based on the types of notes in the
    # match snotes (only integers)
    divs_arg = [
        max(int((beat_type_map_from_beats(note.OnsetInBeats) / 4)), 1)
        * note.Offset.denominator
        * (note.Offset.tuple_div or 1)
        for note in snotes
    ]
    divs_arg += [
        max(int((beat_type_map_from_beats(note.OnsetInBeats) / 4)), 1)
        * note.Duration.denominator
        * (note.Duration.tuple_div or 1)
        for note in snotes
    ]
    # time signatures in omitted sections need to be considered as well
    divs_arg += [max(int(tsg.denominator / 4), 1) for _, _, tsg in ts]

    onset_in_beats = np.array([note.OnsetInBeats for note in snotes])
    unique_onsets, inv_idxs = np.unique(onset_in_beats, return_inverse=True)

    onset_in_quarters = beats_to_quarters(unique_onsets)
    iois_in_quarters = np.diff(onset_in_quarters)

    # ___ these divs are relative to quarters;
    divs = np.lcm.reduce(np.unique(divs_arg))
    onset_in_divs = np.r_[0, np.cumsum(np.round(divs * iois_in_quarters), dtype=int)][
        inv_idxs
    ]
    onset_in_quarters = onset_in_quarters[inv_idxs]

    part.set_quarter_duration(0, divs)

    # position of the first note and start of the score in quarters
    t = float(beats_to_quarters(min_time))
    offset = min(
        [t, 0]
        + [float(beats_to_quarters(s.StartInBeatsUnfolded)) for s in omitted_sections]
    )
    onset_in_divs += int(round((t - offset) * divs))

    # start of the bar according to each note
    note_bar_times = []
    for note, note_onset in zip(snotes, onset_in_quarters):
        beat_type = beat_type_map_from_beats(note.OnsetInBeats)
        bar_offset = (note.Beat - 1) * 4 / beat_type
        on_off_scale = 1
        if not match_offset_duration_in_whole:
            on_off_scale = beat_type
        beat_offset = (
            4
            / on_off_scale
            * note.Offset.numerator
            / (note.Offset.denominator * (note.Offset.tuple_div or 1))
        )
        note_bar_times.append(float(note_onset - bar_offset - beat_offset))

    # list of (start in quarters, measure number) for each bar, and the index
    # of the bar of each note.
    bars, bar_idxs = get_bars_from_snotes(
        snotes=snotes,
        onset_in_quarters=onset_in_quarters,
        note_bar_times=note_bar_times,
        bar_duration=lambda x: (
            beats_map_from_quarters(x) * 4 / beat_type_map_from_quarters(x)
        ),
    )

    # sort notes by bar (if there are repeated measure numbers, the
    # notes in these measures are not in order).
    bar_order = np.argsort(bar_idxs, kind="stable")
    snotes = [snotes[i] for i in bar_order]
    onset_in_divs = onset_in_divs[bar_order]
    bar_idxs = bar_idxs[bar_order]

    def position_in_divs(bar: int, beat_time: float) -> int:
        # Position of a time or key signature. Start of the (closest) bar with
        # this number, or the position in beats if there are no notes in this bar
        time = float(beats_to_quarters(beat_time))
        bar_times = [bar_time for bar_time, bar_name in bars if bar_name == bar]
        if len(bar_times) > 0:
            time = min(bar_times, key=lambda x: abs(x - time))
        return max(0, int(round(divs * (time - offset))))

    for ni, note in enumerate(snotes):
        # start of bar in quarter units
        bar_start = bars[bar_idxs[ni]][0]

        on_off_scale = 1
        # on_off_scale = 1 means duration and beat offset are given in
        # whole notes, else they're given in beats (as in the KAIST data)
        if not match_offset_duration_in_whole:
            on_off_scale = beat_type_map_from_quarters(bar_start)

        # offset within bar in quarter units adjusted for different
        # time signatures -> 4 / beat_type_map_from_quarters(bar_start)
        bar_offset = (note.Beat - 1) * 4 / beat_type_map_from_quarters(bar_start)

        # offset within beat in quarter units adjusted for different
        # time signatures -> 4 / beat_type_map_from_quarters(bar_start)
        beat_offset = (
            4
            / on_off_scale
            * note.Offset.numerator
            / (note.Offset.denominator * (note.Offset.tuple_div or 1))
        )

        # convert the onset time in quarters (0 at first barline) to onset
        # time in divs (0 at first note)
        onset_divs = int(round(divs * (bar_start + bar_offset + beat_offset - offset)))

        if not np.isclose(onset_divs, onset_in_divs[ni], atol=divs * 0.01):
            warnings.warn(
                "Calculated `onset_divs` does not match `OnsetInBeats` " "information!."
            )
            onset_divs = onset_in_divs[ni]

        assert onset_divs >= 0
        assert np.isclose(onset_divs, onset_in_divs[ni], atol=divs * 0.01)
        is_tied = False
        articulations = set()
        if "staccato" in note.ScoreAttributesList or "stac" in note.ScoreAttributesList:
            articulations.add("staccato")
        if "accent" in note.ScoreAttributesList:
            articulations.add("accent")
        if "leftOutTied" in note.ScoreAttributesList:
            is_tied = True

        # dictionary with keyword args with which the Note
        # (or GraceNote) will be instantiated
        note_attributes = dict(
            step=note.NoteName,
            octave=note.Octave,
            alter=note.Modifier,
            id=note.Anchor,
            articulations=articulations,
            technical=[],
        )

        staff_nr = next(
            (a[-1] for a in note.ScoreAttributesList if a.startswith("staff")), None
        )
        try:
            note_attributes["staff"] = int(staff_nr)
        except (TypeError, ValueError):
            # no staff attribute, or staff attribute does not end with a number
            note_attributes["staff"] = None

        if "s" in note.ScoreAttributesList:
            note_attributes["voice"] = 1
        elif any(a.startswith("v") for a in note.ScoreAttributesList):
            note_attributes["voice"] = next(
                (
                    int(a[1:])
                    for a in note.ScoreAttributesList
                    if vnumber_pattern.match(a)
                ),
                None,
            )
        else:
            note_attributes["voice"] = next(
                (int(a) for a in note.ScoreAttributesList if number_pattern.match(a)),
                None,
            )

        if any(a.startswith("fingering") for a in note.ScoreAttributesList):
            note_attributes["technical"].append(
                next(
                    (
                        score.Fingering(int(a[9:]))
                        for a in note.ScoreAttributesList
                        if fingering_pattern.match(a)
                    ),
                    None,
                )
            )

        # get rid of this if as soon as we have a way to iterate over the
        # duration components. For now we have to treat the cases simple
        # and compound durations separately.
        if note.Duration.add_components:
            prev_part_note = None

            for i, (num, den, tuple_div) in enumerate(note.Duration.add_components):
                # when we add multiple notes that are tied, the first note will
                # get the original note id, and subsequent notes will get a
                # derived note id (by appending, 'a', 'b', 'c',...)
                if i > 0:
                    # tnote_id = 'n{}_{}'.format(note.Anchor, i)
                    note_attributes["id"] = score._make_tied_note_id(
                        note_attributes["id"]
                    )

                part_note = score.Note(**note_attributes)

                # duration_divs from local beats --> 4/beat_type_map(bar_start)

                duration_divs = int(
                    (4 / on_off_scale) * divs * num / (den * (tuple_div or 1))
                )

                assert duration_divs > 0
                offset_divs = onset_divs + duration_divs
                part.add(part_note, onset_divs, offset_divs)

                if prev_part_note:
                    prev_part_note.tie_next = part_note
                    part_note.tie_prev = prev_part_note
                prev_part_note = part_note
                onset_divs = offset_divs

        else:
            num = note.Duration.numerator
            den = note.Duration.denominator
            tuple_div = note.Duration.tuple_div

            # duration_divs from local beats --> 4/beat_type_map(bar_start)
            duration_divs = int(
                divs * 4 / on_off_scale * num / (den * (tuple_div or 1))
            )
            offset_divs = onset_divs + duration_divs

            # notes with duration 0, are also treated as grace notes, even if
            # they do not have a 'grace' score attribute
            if "grace" in note.ScoreAttributesList or note.Duration.numerator == 0:
                part_note = score.GraceNote(
                    grace_type="appoggiatura", **note_attributes
                )

            else:
                part_note = score.Note(**note_attributes)
            part.add(part_note, onset_divs, offset_divs)
        # Check if the note is tied and if so, add the tie information
        if is_tied:
            found = False
            # iterate over all notes in the Timeline that end at the starting point.
            for el in part_note.start.iter_ending(score.Note):
                if isinstance(el, score.Note):
                    condition = (
                        el.step == note_attributes["step"]
                        and el.octave == note_attributes["octave"]
                        and el.alter == note_attributes["alter"]
                    )
                    if condition:
                        el.tie_next = part_note
                        part_note.tie_prev = el
                        found = True
                        break
            if not found:
                warnings.warn(
                    "Tie information found, but no previous note found to tie to for note {}.".format(
                        part_note.id
                    )
                )
    # add time signatures
    for i, (ts_beat_time, ts_bar, tsg) in enumerate(ts):
        ts_beats = tsg.numerator
        ts_beat_type = tsg.denominator
        ts_divs = position_in_divs(ts_bar, ts_beat_time)
        if i == 0 and ts_bar not in [bar_name for _, bar_name in bars]:
            # first bar has no notes
            ts_divs = 0
        part.add(score.TimeSignature(ts_beats, ts_beat_type), ts_divs)
    # add key signatures
    for i, (ks_beat_time, ks_bar, keys) in enumerate(mf.key_signatures):
        ks_divs = position_in_divs(ks_bar, ks_beat_time)
        if i == 0 and ks_bar not in [bar_name for _, bar_name in bars]:
            ks_divs = 0
        # TODO
        # * use key estimation if there are multiple defined keys
        # fifths, mode = key_name_to_fifths_mode(key_name)
        part.add(score.KeySignature(keys.fifths, keys.mode), ks_divs)

    add_staffs(part)
    # add_clefs(part)

    # add bars without notes. The measure numbers of time and key signatures
    # at the start of a bar are known, even if there are no notes in this bar
    signature_bars = [
        (float(beats_to_quarters(line.TimeInBeats)), int(line.Measure))
        for line in mf.time_sig_lines + mf.key_sig_lines
        if hasattr(line, "Beat") and line.Beat == 1 and float(line.Offset) == 0
    ]
    bars = add_empty_bars(
        bars=bars,
        signature_bars=signature_bars,
        ts=ts,
        beats_to_quarters=beats_to_quarters,
        start=offset,
        end=max(
            [float(beats_to_quarters(s.EndInBeatsUnfolded)) for s in omitted_sections],
            default=None,
        ),
    )

    measures = []
    for barline_in_quarters, measure_name in bars:
        barline_in_divs = int(round(divs * (barline_in_quarters - offset)))
        if barline_in_divs < 0:
            barline_in_divs = 0
        if len(measures) > 0:
            part.add(measures[-1], None, barline_in_divs)
        measures.append(score.Measure(name=str(measure_name)))
        part.add(measures[-1], barline_in_divs)
    last_closing_barline = barline_in_divs + int(
        round(
            divs
            * beats_map_from_quarters(barline_in_quarters)
            * 4
            / beat_type_map_from_quarters(barline_in_quarters)
        )
    )
    part.add(measures[-1], None, last_closing_barline)

    # add omitted sections
    for section in omitted_sections:
        section_start, section_end = (
            max(0, int(round(divs * (float(beats_to_quarters(beats)) - offset))))
            for beats in (section.StartInBeatsUnfolded, section.EndInBeatsUnfolded)
        )
        part.add(
            score.OmittedSection(
                id=section.id,
                attributes=[a for a in section.SectionAttrList if a != ""],
            ),
            section_start,
            section_end,
        )

    add_rests_in_empty_bars(part, measures)

    # add the rest of the measures automatically
    score.add_measures(part)

    # `add_measures` numbers the measures consecutively, but repeated measures
    # should have the same number (which always starts with 1).
    first_measure_name = min(measure_name for _, measure_name in bars)
    for measure, (_, measure_name) in zip(measures, bars):
        measure.number = int(measure_name - first_measure_name + 1)

    score.tie_notes(part)
    score.find_tuplets(part)

    n_voices = set([n.voice for n in part.notes])
    if len(n_voices) == 1 and None in n_voices:
        for note in part.notes_tied:
            if note.voice is None:
                note.voice = 1
    elif len(n_voices) > 1 and None in n_voices:
        n_voices.remove(None)
        for note in part.notes_tied:
            if note.voice is None:
                note.voice = max(n_voices) + 1

    return part


def make_timesig_maps(
    ts_orig: List[Tuple[float, int, MatchTimeSignature]],
    max_time: float,
) -> Tuple[Callable, Callable, Callable, Callable, float, float]:
    """
    Create time signature (interpolation) maps

    Parameters
    ----------
    ts_orig : List[Tuple[float, int, MatchTimeSignature]]
        A list of tuples containing position in beats, measure and
        MatchTimeSignature instances
    max_time : float
        Maximal time of the time signatures

    Returns
    -------
    beats_map: callable
    qbeats_map: callable
    beat_type_map: callable
    qbeat_type_map: callable
    start_q: float
    end_q: float
    """
    # TODO: make sure that ts_orig covers range from min_time
    # return two functions that map score times (in quarter units) to time sig
    # beats, and time sig beat_type respectively
    ts = list(ts_orig)
    assert len(ts) > 0
    ts.append((max_time, None, ts[-1][2]))

    x = np.array([t for t, _, _ in ts])
    y = np.array([(x.numerator, x.denominator) for _, _, x in ts])

    start_q = x[0] * 4 / y[0, 1]
    x_q = np.cumsum(np.r_[start_q, 4 * np.diff(x) / y[:-1, 1]])
    end_q = x_q[-1]

    # TODO: fix error with bounds
    qbeats_map = interp1d(
        x_q,
        y[:, 0],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 0], y[-1, 0]),
    )
    qbeat_type_map = interp1d(
        x_q,
        y[:, 1],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 1], y[-1, 1]),
    )
    beats_map = interp1d(
        x,
        y[:, 0],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 0], y[-1, 0]),
    )
    beat_type_map = interp1d(
        x,
        y[:, 1],
        kind="previous",
        bounds_error=False,
        fill_value=(y[0, 1], y[-1, 1]),
    )

    return beats_map, qbeats_map, beat_type_map, qbeat_type_map, start_q, end_q


def make_beats_to_quarters_map(
    ts: List[Tuple[float, int, MatchTimeSignature]],
) -> Callable:
    """
    Create a map from score time in beats to score time in quarters.

    Parameters
    ----------
    ts : List[Tuple[float, int, MatchTimeSignature]]
        A list of tuples containing position in beats, measure and
        MatchTimeSignature instances

    Returns
    -------
    beats_to_quarters : callable
    """
    ts_beats = np.array([t for t, _, _ in ts], dtype=float)
    quarters_per_beat = np.array([4 / tsg.denominator for _, _, tsg in ts])
    ts_quarters = np.cumsum(
        np.r_[
            ts_beats[0] * quarters_per_beat[0],
            np.diff(ts_beats) * quarters_per_beat[:-1],
        ]
    )

    def beats_to_quarters(beats: Union[float, np.ndarray]) -> np.ndarray:
        idx = np.searchsorted(ts_beats, beats, side="right") - 1
        idx = np.clip(idx, 0, None)
        return ts_quarters[idx] + (beats - ts_beats[idx]) * quarters_per_beat[idx]

    return beats_to_quarters


def get_bars_from_snotes(
    snotes: List[BaseSnoteLine],
    onset_in_quarters: np.ndarray,
    note_bar_times: List[float],
    bar_duration: Callable[[float], float],
) -> Tuple[List[Tuple[float, int]], np.ndarray]:
    """
    Get the bars which contain score notes.

    Parameters
    ----------
    snotes : List[BaseSnoteLine]
        The score notes
    onset_in_quarters : np.ndarray
        The score onset of each note in quarters.
    note_bar_times : List[float]
        The start (in quarters) of the bar of each note, according to the
        onset and the position in the bar of this note.
    bar_duration : Callable
        A map from score time in quarters to the (nominal) duration
        of the bar at this time in quarters.

    Returns
    -------
    bars : List[Tuple[float, int]]
        A list of tuples with the start (in quarters) and the measure number of
        each bar, sorted by time.
    bar_idxs : np.ndarray
        The index of the bar of each note.

    Notes
    -----
    * In unfolded scores the same measure number can occur several times. The
      notes of these measures are told apart by the start of the bar.
    """
    notes_in_measure = defaultdict(list)
    for ni, note in enumerate(snotes):
        notes_in_measure[note.Measure].append(ni)

    # tuples of bar start, measure number, notes in this bar
    bars = []
    for measure, note_idxs in sorted(notes_in_measure.items()):
        # split notes into candidates for a repetition of the measure
        by_time = sorted(note_idxs, key=lambda ni: note_bar_times[ni])
        occurrences = [[by_time[0]]]
        for prev, ni in zip(by_time[:-1], by_time[1:]):
            time, prev_time = note_bar_times[ni], note_bar_times[prev]
            if time - prev_time >= 0.5 * bar_duration(prev_time):
                occurrences.append([])
            occurrences[-1].append(ni)

        # If there are notes of a candidate that sound after the next one
        # started, these are not repetitions but the result of imprecise onsets.
        if any(
            max(onset_in_quarters[ni] for ni in current)
            > min(note_bar_times[ni] for ni in following)
            for current, following in zip(occurrences[:-1], occurrences[1:])
        ):
            occurrences = [note_idxs]

        for occurrence in occurrences:
            # the first note in a bar determines its start
            first = min(occurrence)
            bars.append((note_bar_times[first], int(measure), occurrence))

    bars.sort(key=lambda bar: bar[:2])

    bar_idxs = np.empty(len(snotes), dtype=int)
    for bar_idx, (_, _, note_idxs) in enumerate(bars):
        bar_idxs[note_idxs] = bar_idx

    return [bar[:2] for bar in bars], bar_idxs


def add_empty_bars(
    bars: List[Tuple[float, int]],
    signature_bars: List[Tuple[float, int]],
    ts: List[Tuple[float, int, MatchTimeSignature]],
    beats_to_quarters: Callable,
    start: float,
    end: Optional[float] = None,
) -> List[Tuple[float, int]]:
    """
    Add bars without notes (e.g., bars in omitted sections), which are missing
    from a list of bars.

    Parameters
    ----------
    bars : List[Tuple[float, int]]
        A list of tuples with the start (in quarters) and the measure number of
        each bar, sorted by time.
    signature_bars : List[Tuple[float, int]]
        A list of tuples with the start (in quarters) and the measure number of
        the bars in which there is a time or key signature.
    ts : List[Tuple[float, int, MatchTimeSignature]]
        A list of tuples containing position in beats, measure and
        MatchTimeSignature instances
    beats_to_quarters : Callable
        A map from score time in beats to score time in quarters.
    start : float
        Start of the score in quarters.
    end : float, optional
        Bars are added after the last bar until this time (in quarters).

    Returns
    -------
    all_bars : List[Tuple[float, int]]
        A list of bars including bars without notes.

    Notes
    -----
    * There are no bars missing between bars with consecutive measure numbers,
      irrespective of their duration (which can be irregular).
    * If the measure number does not increase (i.e., at a repeat), the number
      of missing bars is determined by their duration, and they are numbered as
      a continuation of the last bar before the repeat.
    """
    eps = 1e-3
    bars = list(bars)
    ts_times = np.array([float(beats_to_quarters(t)) for t, _, _ in ts])
    bar_durations = [4 * tsg.numerator / tsg.denominator for _, _, tsg in ts]

    def previous_barline(time: float) -> float:
        # there is always a barline where the time signature changes
        idx = max(np.searchsorted(ts_times, time - eps) - 1, 0)
        barline = time - bar_durations[idx]
        if barline < ts_times[idx] < time - eps:
            barline = ts_times[idx]
        return float(barline)

    def next_barline(time: float) -> float:
        idx = max(np.searchsorted(ts_times, time + eps) - 1, 0)
        barline = time + bar_durations[idx]
        if idx + 1 < len(ts_times):
            barline = min(barline, ts_times[idx + 1])
        return float(barline)

    # add bars of time and key signatures, if they fit in between the other bars
    for bar_time, bar_name in sorted(set(signature_bars)):
        if bar_time <= start + eps or (end is None and bar_time > bars[-1][0]):
            continue
        idx = int(np.searchsorted([time for time, _ in bars], bar_time))
        prev_time, prev_name = bars[idx - 1] if idx > 0 else (start, -np.inf)
        next_time, next_name = bars[idx] if idx < len(bars) else (end, np.inf)
        if not prev_time + eps < bar_time < next_time - eps:
            continue
        after_prev, before_next = prev_name < bar_name, bar_name < next_name
        # at a repeat, the bar can be on either side of the repeat sign
        is_repeat = next_name <= prev_name
        if (after_prev and before_next) or (is_repeat and (after_prev or before_next)):
            bars.insert(idx, (bar_time, bar_name))

    all_bars = []

    # bars before the first bar
    first_time, first_name = bars[0]
    barlines = []
    barline = first_time
    while barline > start + eps:
        barline = max(previous_barline(barline), start)
        barlines.insert(0, barline)
    first_name -= len(barlines)
    if first_name < 0:
        # the first bar is already a repetition
        first_name = 1
    all_bars += [(barline, first_name + i) for i, barline in enumerate(barlines)]

    # Bars between bars. Barlines are computed backwards, since
    # the duration of the last bar before the gap can be irregular.
    for (bar_time, bar_name), (next_time, next_name) in zip(bars[:-1], bars[1:]):
        all_bars.append((bar_time, bar_name))
        n_missing = next_name - bar_name - 1 if next_name > bar_name else np.inf
        barlines = []
        barline = previous_barline(next_time)
        while barline > bar_time + eps and len(barlines) < n_missing:
            barlines.insert(0, barline)
            barline = previous_barline(barline)
        all_bars += [(barline, bar_name + i + 1) for i, barline in enumerate(barlines)]

    # bars after the last bar
    bar_time, bar_name = bars[-1]
    all_bars.append((bar_time, bar_name))
    if end is not None:
        bar_time = next_barline(bar_time)
        while bar_time < end - eps:
            bar_name += 1
            all_bars.append((bar_time, bar_name))
            bar_time = next_barline(bar_time)

    return all_bars


def add_rests_in_empty_bars(part: Part, measures: List[score.Measure]) -> None:
    """
    Add rests (in place) to measures without notes, and before the first note
    if it is not at the start of a measure. Omitted sections are not rests.

    Parameters
    ----------
    part : Part
        The part
    measures : List[score.Measure]
        The measures of the part
    """
    note_times = np.array(
        [
            (note.start.t, note.end.t)
            for note in part.iter_all(score.GenericNote, include_subclasses=True)
        ]
    )
    first_onset = note_times[:, 0].min()
    omitted_times = [
        (section.start.t, section.end.t) for section in part.omitted_sections
    ]

    for measure in measures:
        start, end = measure.start.t, measure.end.t

        if start <= first_onset < end:
            rest_start = max(
                [start]
                + [o_end for o_start, o_end in omitted_times if o_start < first_onset]
            )
            if rest_start < first_onset:
                part.add(score.Rest(voice=1, staff=1), rest_start, first_onset)

        # notes with no duration are grace notes
        has_notes = np.any(
            (note_times[:, 0] < end)
            & ((note_times[:, 1] > start) | (note_times[:, 0] >= start))
        )
        is_omitted = any(
            o_start < end and o_end > start for o_start, o_end in omitted_times
        )
        if start < end and not has_notes and not is_omitted:
            for staff in (1, 2):
                part.add(score.Rest(voice=staff, staff=staff), start, end)


def add_staffs(part: Part, split: int = 55, only_missing: bool = True) -> None:
    """
    Method to add staff information to a part

    Parameters
    ----------
    part: Part
        Part to add staff information to.
    split: int
        MIDI pitch to split staff into upper and lower. Default is 55
    only_missing: bool
        If True, only add staff to those notes that do not have staff info already.
    """
    # assign staffs using a hard limit
    notes = part.notes_tied
    for n in notes:
        if only_missing and n.staff:
            continue

        if n.midi_pitch > split:
            staff = 1
        else:
            staff = 2

        n.staff = staff

        n_tied = n.tie_next
        while n_tied:
            n_tied.staff = staff
            n_tied = n_tied.tie_next

    part.add(score.Clef(staff=1, sign="G", line=2, octave_change=0), 0)
    part.add(score.Clef(staff=2, sign="F", line=4, octave_change=0), 0)


def validate_match_ids(mf):
    """
    Check a matchfile for duplicate snote and note IDs.

    This function will:
    - remove all deletions with a score ID that occurs in multiple lines.
    - remove all insertions with a performance ID that occurs in multiple lines.
    - warn about virtual notes that point to notes that are not in the match file.

    Virtual notes are pointers to other notes, and thus not duplicates.

    Handles cases with conflicting match/insertion(s) and match/deletion(s) tuples
    with any number of insertions or deletions and a single match by keeping
    only the match.

    Unhandled cases:
    - multiple conflicting matches: all are kept.
    - multiple insertions with the same ID: all are deleted.
    - multiple deletions with the same ID: all are deleted.

    Parameters
    ----------
    mf: MatchFile
        MatchFile to validate

    Returns
    -------
    Updates the representation of the matchfile by removing match lines.
    """

    # Check if the matchfile is valid (i.e. check for snote duplicates)
    sids = np.array([n.Anchor for n in mf.snotes])
    # First check if score ids are unique
    sids_unique, counts = np.unique(sids, return_counts=True)
    sids_to_check = sids_unique[np.where(counts > 1)[0]]
    if len(sids_to_check) > 0:
        indices_to_del = []
        for i, line in enumerate(mf.lines):
            if isinstance(line, BaseDeletionLine):
                if line.Anchor in sids_to_check:
                    indices_to_del.append(i)
        warnings.warn(
            "Matchfile contains duplicate score notes. "
            "Removing {} deletions.".format(len(indices_to_del))
        )
        mf.lines = np.delete(mf.lines, indices_to_del)

    # Check if the matchfile is valid (i.e. check for performance note duplicates)
    pids = np.array([n.Id for n in mf.notes])
    pids_unique, counts = np.unique(pids, return_counts=True)
    pids_to_check = pids_unique[np.where(counts > 1)[0]]
    if len(pids_to_check) > 0:
        indices_to_del = []
        for i, line in enumerate(mf.lines):
            if isinstance(line, BaseInsertionLine):
                if line.Id in pids_to_check:
                    indices_to_del.append(i)
        warnings.warn(
            "Matchfile contains duplicate performance notes. "
            "Removing {} insertions.".format(len(indices_to_del))
        )
        mf.lines = np.delete(mf.lines, indices_to_del)

    # Check if the virtual notes point to existing notes
    note_pairs = [line for line in mf.lines if isinstance(line, note_pair_classes)]
    virtual_sids = set(
        line.snote.Anchor
        for line in note_pairs
        if isinstance(line.snote, VirtualSnoteLine)
    )
    virtual_pids = set(
        format_pnote_id(line.note.Id)
        for line in note_pairs
        if isinstance(line.note, VirtualNoteLine)
    )
    sids = set(n.Anchor for n in mf.snotes)
    pids = set(format_pnote_id(n.Id) for n in mf.notes)
    for note_type, missing_ids in (
        ("score", virtual_sids - sids),
        ("performance", virtual_pids - pids),
    ):
        if len(missing_ids) > 0:
            warnings.warn(
                f"Matchfile contains virtual {note_type} notes pointing to notes "
                f"that are not in the matchfile: {', '.join(sorted(missing_ids))}."
            )

    # Check if the virtual notes point to notes that are aligned
    deleted_sids = set(
        line.snote.Anchor for line in mf.lines if isinstance(line, BaseDeletionLine)
    )
    inserted_pids = set(
        format_pnote_id(line.note.Id)
        for line in mf.lines
        if isinstance(line, BaseInsertionLine)
    )
    for note_type, not_aligned_ids in (
        ("score", virtual_sids & deleted_sids),
        ("performance", virtual_pids & inserted_pids),
    ):
        if len(not_aligned_ids) > 0:
            warnings.warn(
                f"Matchfile contains virtual {note_type} notes pointing to "
                f"{'deletions' if note_type == 'score' else 'insertions'}: "
                f"{', '.join(sorted(not_aligned_ids))}."
            )


if __name__ == "__main__":
    pass
